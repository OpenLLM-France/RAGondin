import re
import asyncio
from .llm import LLM
from typing import Optional
from abc import ABCMeta, abstractmethod, ABC
from langchain_openai import ChatOpenAI
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain.callbacks import StdOutCallbackHandler
from loguru import logger
from langchain_core.runnables import RunnableLambda
import openai

class BaseChunker(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def split_document(self, doc: Document):
        pass

    



templtate = """
<document>
{origin}
</document>

This is the chunk we want to situate within the document named `{source}`.
The document’s name itself may contain relevant information (such as "employees CV," "tutorials," etc.) that can help contextualize the content. 

<chunk>
{chunk}
</chunk>

Please provide a brief, succinct context to situate this chunk within the document, specifically to improve search retrieval of this chunk. 
Respond only with the concise context in the same language as the provided document and chunk.
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
        # We can apply apply this function 'split_text' once.
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
        filtered_chunks = []
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
                filtered_chunks.append(chunk)
        
        return filtered_chunks


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
        self.context_generator = (prompt | llm | StrOutputParser())
    

    def contextualize(self, idxs: list, b_chunks: list[Document], pages: list[Document], source: str):
        # TODO: consider adding previous paragraph
        if not self.contextual_retrieval:
            return [chunk.page_content for chunk in b_chunks]
        
        origin = f"""first page: {pages[0]}\n\nlast page: {pages[-1]}"""
        try:
            b_contexts = self.context_generator\
                .with_retry(
                    retry_if_exception_type=(Exception, ),
                    wait_exponential_jitter=True,
                    stop_after_attempt=3
                )\
                .batch(
                    [
                        {
                            "origin": origin, 
                            'chunk': chunk.page_content,'source': source
                        } for idx, chunk in zip(idxs, b_chunks)
                    ], 
                    # config={"callbacks": [StdOutCallbackHandler()]}
                )

            s = """chunk's context: {chunk_context}\n\n=> chunk: {chunk}"""
            return [s.format(chunk=chunk.page_content, chunk_context=chunk_context) for chunk, chunk_context in zip(b_chunks, b_contexts)]
        except Exception as e:
            logger.warning(f"An error happened with document `{source}`: {e}")
            return [chunk.page_content for chunk in b_chunks]


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

        i = 0
        filtered_chunks = []
        batch_size = 4

        for j in range(0, len(chunks), batch_size):
            b_chunks = chunks[j:j+batch_size] 
            idxs = [idx for idx in range(j, min(j+batch_size, len(chunks))) ]
            
            b_chunks_w_context = self.contextualize(
                idxs=idxs, b_chunks=b_chunks, 
                pages=pages, source=source
            )

            for chunk, b_chunk_w_context in zip(b_chunks, b_chunks_w_context):
                start_idx = chunk.metadata["start_index"]

                while not (page_idx[i]["start_idx"] <= start_idx <= page_idx[i]["end_idx"]) and i < len(page_idx)-1:
                    i += 1
            
                if len(chunk.page_content.strip()) > 1:
                    chunk.page_content = b_chunk_w_context
                    chunk.metadata = {
                        "page": page_idx[i]["page"], 
                        "source": source,
                    }
                    filtered_chunks.append(chunk)

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