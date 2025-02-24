import re
from ..llm import LLM
from typing import Optional
from abc import abstractmethod, ABC
from langchain_openai import ChatOpenAI
from langchain_core.documents.base import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain.callbacks import StdOutCallbackHandler
from loguru import logger
from langchain_core.runnables import RunnableLambda
from omegaconf import OmegaConf


class ABCChunker(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def split_document(self, doc: Document):
        pass


# templtate = """
# <document>
# {origin}
# </document>

# This is the chunk we want to situate within the document named `{source}`.
# The document’s name itself may contain relevant information (such as "employees CV," "tutorials," etc.) that can help contextualize the content. 

# <chunk>
# {chunk}
# </chunk>

# Please provide a brief, succinct context to situate this chunk within the document, specifically to improve search retrieval of this chunk. 
# Respond only with the concise context in the same language as the provided document and chunk.
# """

templtate = """
<document>
Title: {source}  
# The document title may contain key metadata (e.g., "cv", "videos", "client proposals").
First page of the document: {first_page}  
Previous chunk: {prev_chunk}  
</document>

<current_chunk>
{chunk}
</current_chunk>

**Task:**  
Provide a concise, one-sentence context that situates the *<current_chunk>* within the *<document>*, integrating relevant information from:  
1. **Title** (e.g., type, or category information encoded in the filename).  
2. **First page of the document**.  
3. **Previous chunk**.  
4. **Current chunk content**.  

**Response Format:**  
- Only provide a single, concise contextual sentence for the *<current_chunk>*.  
- Write the response in the **same language** as the current chunk to enhance retrieval quality.  
- Do not include any additional text or explanation.  
"""


class ChunkContextualizer:
    def __init__(self, contextual_retrieval: bool = False, llm: Optional[ChatOpenAI] = None):
        self.contextual_retrieval = contextual_retrieval
        self.context_generator = None
        if self.contextual_retrieval:
            assert isinstance(llm, ChatOpenAI), f'The `llm` should be of type `ChatOpenAI` is contextual_retrieval is `True`'
            prompt = ChatPromptTemplate.from_template(
                template=templtate
            )
            self.context_generator = (prompt | llm | StrOutputParser())

    def contextualize(self, prev_chunks: list[Document], b_chunks: list[Document], pages: list[str], source: str) -> list[str]:
        if not self.contextual_retrieval:
            return [chunk.page_content for chunk in b_chunks]
        
        try:
            b_contexts = self.context_generator\
                .with_retry(
                    retry_if_exception_type=(Exception,),
                    wait_exponential_jitter=False,
                    stop_after_attempt=3
                )\
                .batch([
                    {
                        "first_page": pages[0], 
                        'prev_chunk': prev_chunk.page_content,
                        'chunk': chunk.page_content,
                        'source': source
                    } for prev_chunk, chunk in zip(prev_chunks, b_chunks)
                ])

            s = "chunk's context: {chunk_context}\n\n=> chunk: {chunk}"
            return [s.format(chunk=chunk.page_content, chunk_context=chunk_context) 
                    for chunk, chunk_context in zip(b_chunks, b_contexts)]
        
        except Exception as e:
            logger.warning(f"An error occurred with document `{source}`: {e}")
            return [chunk.page_content for chunk in b_chunks]



class RecursiveSplitter(ABCChunker):
    def __init__(
            self, 
            chunk_size: int=200, 
            chunk_overlap: int=20, 
            contextual_retrieval: bool=True,
            llm: Optional[ChatOpenAI]=None,
            **args
        ):
        super().__init__(contextual_retrieval=contextual_retrieval, llm=llm)

        from langchain.text_splitter import RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **args
        )
        self.contextualizer = ChunkContextualizer(contextual_retrieval=contextual_retrieval, llm=llm)


    def split_document(self, doc: Document):
        text = ''
        page_idx = []
        source = doc.metadata["source"]
        page_sep = doc.metadata["page_sep"]
        pages: list[str] = doc.page_content.split(sep=page_sep)
        start_index = 0
        
        # TODO: We can apply apply this function 'split_text' once.
        for page_num, p in enumerate(pages, start=1):
            text += '\n' + p
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
        batch_size = 4
        filtered_chunks = []

        chunks = [Document(page_content='')] + chunks

        for j in range(1, len(chunks), batch_size):
            b_chunks = chunks[j:j+batch_size]
            prev_chunks = chunks[j-1:j+batch_size-1]

            b_chunks_w_context = self.contextualizer.contextualize(
                prev_chunks=prev_chunks, b_chunks=b_chunks, 
                pages=pages, source=source
            )

            for chunk, b_chunk_w_context in zip(b_chunks, b_chunks_w_context):
                start_idx = full_text_preprocessed.find(chunk.page_content)

                while not (page_idx[i]["start_idx"] <= start_idx <= page_idx[i]["end_idx"]) and i < len(page_idx)-1:
                    i += 1
            
                if len(chunk.page_content.strip()) > 1:
                    chunk.page_content = b_chunk_w_context
                    metadata = doc.metadata
                    metadata.update({"page": page_idx[i]["page"]})
                    chunk.metadata = metadata
                    filtered_chunks.append(chunk)

        return filtered_chunks


class SemanticSplitter(ABCChunker):
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
        self.contextualizer = ChunkContextualizer(contextual_retrieval=contextual_retrieval, llm=llm)

    def split_document(self, doc: Document):
        text = ''
        page_idx = []
        metadata = doc.metadata
        source = metadata["source"]
        page_sep = metadata.pop("page_sep")
        pages = doc.page_content.split(sep=page_sep) 

        start_index = 0
        for page_num, page_txt in enumerate(pages, start=1):
            text += '\n' + page_txt
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
        chunks = [Document(page_content='')] + chunks

        for j in range(1, len(chunks), batch_size):
            b_chunks = chunks[j:j+batch_size]
            prev_chunks = chunks[j-1:j+batch_size-1]
            b_chunks_w_context = self.contextualizer.contextualize(
                prev_chunks=prev_chunks, b_chunks=b_chunks, 
                pages=pages, source=source
            )

            for chunk, b_chunk_w_context in zip(b_chunks, b_chunks_w_context):
                start_idx = chunk.metadata["start_index"]

                while not (page_idx[i]["start_idx"] <= start_idx <= page_idx[i]["end_idx"]):
                    i += 1
            
                # print(page_idx[i], start_idx)
                if len(chunk.page_content.strip()) > 1:
                    chunk.page_content = b_chunk_w_context

                    metadata.update({"page": page_idx[i]["page"]})
                    chunk.metadata = {
                        "page": page_idx[i]["page"],
                        "source": metadata["source"],
                        'sub_url_path': metadata["sub_url_path"]
                    }
                    filtered_chunks.append(chunk)
        return filtered_chunks


class ChunkerFactory:
    CHUNKERS = {
        'recursive_splitter': RecursiveSplitter,
        'semantic_splitter': SemanticSplitter,
    }

    @staticmethod
    def create_chunker(config:OmegaConf, embedder: Optional[HuggingFaceBgeEmbeddings | HuggingFaceEmbeddings]=None) -> ABCChunker:
        # Extract parameters
        chunker_params = OmegaConf.to_container(config.chunker, resolve=True)
        name = chunker_params.pop("name")

        # Initialize and return the chunker
        chunker_class: ABCChunker = ChunkerFactory.CHUNKERS.get(name)
        if not chunker_class:
            raise ValueError(f"Chunker '{name}' is not recognized.")
        
        # Add embeddings if semantic splitter is selected
        if name == 'semantic_splitter':
            if embedder is not None:
                chunker_params.update({"embeddings": embedder})
            else:
                raise AttributeError(f"{name} type chunker requires the `embedder` parameter")

        # Include contextual retrieval if specified
        if chunker_params['contextual_retrieval']:
            chunker_params['llm'] = LLM(config, logger=None).client
        
        return chunker_class(**chunker_params)