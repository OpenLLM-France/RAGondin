from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import AsyncGenerator
from langchain_core.documents.base import Document
import re


class BaseChunker(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def split(self, docs: list[Document]):
        pass


class RecursiveSplitter(BaseChunker):
    def __init__(self, chunk_size: int=200, chunk_overlap: int=20, **args):
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # add_start_index=True,
            **args
        )
    
    def split_doc(self, full_doc: Document):
        text = ''
        page_idx = []
        source = full_doc.metadata["source"]
        page_sep = full_doc.metadata["page_sep"]
        pages: list[str] = full_doc.page_content.split(sep=page_sep)

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
        filtered_chunks = []
        for chunk in chunks:
            start_idx = full_text_preprocessed.find(chunk.page_content)

            while not (page_idx[i]["start_idx"] <= start_idx <= page_idx[i]["end_idx"]) and i < len(page_idx)-1:
                i += 1
            
            # print(i, start_idx, page_idx[i])
            if len(chunk.page_content.strip()) > 1:
                chunk.metadata = {
                    "page": page_idx[i]["page"], 
                    "source": source,
                }
                filtered_chunks.append(chunk)
      
        return filtered_chunks


    async def split(self, batch_docs: AsyncGenerator[list[Document], None]):
        async for b in batch_docs:
            # doc_batch = await asyncio.gather(*[self.split_doc(doc) async for doc in b])
            # yield doc_batch
            yield sum(
                list(map(self.split_doc, b)),
                []
            )

        
class SemanticSplitter(BaseChunker):
    def __init__(self, min_chunk_size: int = 1000, embeddings = None, **args) -> None:
        from langchain_experimental.text_splitter import SemanticChunker

        self.splitter = SemanticChunker(
            embeddings=embeddings, 
            buffer_size=1, 
            breakpoint_threshold_type='percentile', 
            min_chunk_size=min_chunk_size,
            add_start_index=True, 
            **args
        )
    
    def split_doc(self, full_doc: Document):
        text = ''
        page_idx = []
        source = full_doc.metadata["source"]
        page_sep = full_doc.metadata["page_sep"]
        pages = full_doc.page_content.split(sep=page_sep)

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


    async def split(self, batch_docs: AsyncGenerator[list[Document], None]):
        async for b in batch_docs:
            yield sum(
                list(map(self.split_doc, b)),
                []
            )


CHUNKERS = {
    'recursive_splitter': RecursiveSplitter,
    "semantic_splitter": SemanticSplitter,
}