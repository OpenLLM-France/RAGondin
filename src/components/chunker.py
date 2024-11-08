import re
import asyncio
from .llm import LLM
from typing import Optional
from abc import ABCMeta, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain.callbacks import StdOutCallbackHandler
from loguru import logger

class BaseChunker(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    async def split_document(self, doc: Document):
        pass



templtate = """
<document> 
{origin}
</document> 
Here is the chunk we want to situate/contextualize within the whole document named `{source}`. 
The name of the document itself contains relevant info (employees CV, Tutorials, etc.) about the document, so take that into account. 
<chunk> 
{chunk} 
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. 
Answer only with the succinct context and nothing else in the same language as the provided document and chunk.     
"""

prompt = ChatPromptTemplate.from_template(
    template=templtate
)


class RecursiveSplitter(BaseChunker):
    def __init__(
            self, 
            chunk_size: int=200, 
            chunk_overlap: int=20, 
            contextual_retrieval: bool=True,
            llm: Optional[ChatOpenAI]=None,
            **args
        ):

        from langchain.text_splitter import RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **args
        )
        self.contextual_retrieval = contextual_retrieval
        self.gen_context = (prompt | llm | StrOutputParser())

    
    def contextualize(self, chunk: str, pages: str):
        origin = f"""first page: {pages[0]}\n\nlast page: {pages[-1]}"""
        try:
            context = self.gen_context.ainvoke(
                {
                    "origin": origin, 
                    'chunk': chunk
                }
            )
            return context
        except Exception as e:
            print(e)
        

    def split_document(self, doc: Document):
        text = ''
        page_idx = []
        source = doc.metadata["source"]
        page_sep = doc.metadata["page_sep"]
        pages: list[str] = doc.page_content.split(sep=page_sep)


        start_index = 0
        for page_num, p in enumerate(pages, start=1):
            text += ' ' + p
            c = ' '.join(
                self.splitter.split_text(text)
            )
            end_index = len(c) - 1
            page_idx.append(
                {
                    "start_idx": start_index, 
                    "end_idx": end_index, 
                    "page": page_num
                }
            )
            start_index = end_index
                
        # split 
        full_text_preprocessed = ' '.join(
            self.splitter.split_text(text)
        )
        # chunking the full text
        chunks = self.splitter.create_documents([text])

        i = 0
        for chunk in chunks:
            start_idx = full_text_preprocessed.find(chunk.page_content)

            while not (page_idx[i]["start_idx"] <= start_idx <= page_idx[i]["end_idx"]) and i < len(page_idx)-1:
                i += 1
            
            if len(chunk.page_content.strip()) > 1:
                chunk.metadata = {
                    "page": page_idx[i]["page"], 
                    "source": source,
                }
                chunk


class SemanticSplitter(BaseChunker):
    def __init__(self, 
            min_chunk_size: int = 1000, 
            embeddings = None, 
            breakpoint_threshold_amount=85, 
            contextual_retrieval: bool=False, llm: Optional[ChatOpenAI]=None,
            **args
        ) -> None:
        
        from langchain_experimental.text_splitter import SemanticChunker

        self.splitter = SemanticChunker(
            embeddings=embeddings, 
            buffer_size=1, 
            breakpoint_threshold_type='percentile', 
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            min_chunk_size=min_chunk_size,
            add_start_index=True, 
            **args
        )
        self.contextual_retrieval = contextual_retrieval
        self.gen_context = (prompt | llm | StrOutputParser())
    

    async def contextualize(self, b_chunks: list[str], pages: list[Document], source: str):
        origin = f"""first page: {pages[0]}\n\nlast page: {pages[-1]}"""
        try:
            b_contexts = await self.gen_context\
                .with_retry(
                    retry_if_exception_type=(Exception, ),
                    wait_exponential_jitter=False,
                    stop_after_attempt=4
                )\
                .abatch(
                    [{"origin": origin, 'chunk': chunk,'source': source} for chunk in b_chunks], 
                    config={"callbacks": [StdOutCallbackHandler()]}
                )

            s = """chunk's context: {chunk_context}\n\n=> chunk: {chunk}"""
            print(b_contexts)
            return [s.format(chunk=chunk, chunk_context=chunk_context) for chunk, chunk_context in zip(b_chunks, b_contexts)]
        except Exception as e:
            logger.warning(e)
            return [f'{chunk}'.format(chunk=chunk) for chunk in zip(b_chunks)]




    def split_document(self, doc: Document):
        text = ''
        page_idx = []
        source = doc.metadata["source"]
        page_sep = doc.metadata["page_sep"]
        pages = doc.page_content.split(sep=page_sep) 

        start_index = 0
        for page_num, page_txt in enumerate(pages, start=1):
            text += ' ' + page_txt
            c = ' '.join(
                re.split(self.splitter.sentence_split_regex, text)
            )
            end_index = len(c) - 1
            page_idx.append(
                {
                    "start_idx": start_index, 
                    "end_idx": end_index, 
                    "page": page_num
                }
            )
            start_index = end_index

        chunks = self.splitter.create_documents(
            [' '.join(re.split(self.splitter.sentence_split_regex, text))]
        )

        # if self.contextual_retrieval:
        #     bs = 3
        #     CH = []
        #     for i in range(0, len(chunks), bs):
        #         b_chunks = chunks[i:i+bs]
        #         processed_b_chunks = await self.contextualize(b_chunks=b_chunks, pages=pages, source=source)
        #         CH.extend(processed_b_chunks)
            
        #     logger.info("Done")


        i = 0
        filtered_chunks = []
        for semantic_chunk in chunks:
            start_idx = semantic_chunk.metadata["start_index"]
            while not (page_idx[i]["start_idx"] <= start_idx <= page_idx[i]["end_idx"]) and i < len(page_idx)-1:
                i += 1
            
            if len(semantic_chunk.page_content.strip()) > 1:
                semantic_chunk.metadata = {
                    "page": page_idx[i]["page"], 
                    "source": source,
                }
                filtered_chunks.append(semantic_chunk)

        return filtered_chunks


class ChunkerFactory:
    CHUNKERS = {
        'recursive_splitter': RecursiveSplitter,
        'semantic_splitter': SemanticSplitter,
    }


    @staticmethod
    def create_chunker(config, embedder: Optional[HuggingFaceBgeEmbeddings | HuggingFaceEmbeddings]=None) -> BaseChunker:
        # Extract parameters
        chunker_params = dict(config.chunker)
        name = chunker_params.pop("name")
        
        # Convert string values to integers where possible
        for k, v in chunker_params.items():
            try:
                chunker_params[k] = int(v)
            except ValueError:
                pass
        
        # Add embeddings if semantic splitter is selected
        if name == 'semantic_splitter':
            if embedder is not None:
                chunker_params.update({"embeddings": embedder})
            else:
                raise AttributeError(f"{name} type chunker requires the `embedder` parameter")

        # Include contextual retrieval if specified
        chunker_params['contextual_retrieval'] = chunker_params["contextual_retrieval"].lower() == 'true'
        chunker_params['llm'] = LLM(config, logger=None).client
        
        # Initialize and return the chunker
        chunker_class: BaseChunker = ChunkerFactory.CHUNKERS.get(name)
        if not chunker_class:
            raise ValueError(f"Chunker '{name}' is not recognized.")
        return chunker_class(**chunker_params)