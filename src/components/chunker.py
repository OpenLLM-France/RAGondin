from abc import ABCMeta, abstractmethod
import gc
from pathlib import Path
from typing import AsyncGenerator
from langchain_core.documents.base import Document
import re

class BaseChunker(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def split_document(self, doc: Document):
        pass


class RecursiveSplitter(BaseChunker):
    def __init__(self, chunk_size: int=200, chunk_overlap: int=20, **args):
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **args
        )

    

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

class SemanticSplitter(BaseChunker):
    def __init__(self, min_chunk_size: int = 1000, embeddings = None, breakpoint_threshold_amount=85, **args) -> None:
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


CHUNKERS = {
    'recursive_splitter': RecursiveSplitter,
    "semantic_splitter": SemanticSplitter,
}