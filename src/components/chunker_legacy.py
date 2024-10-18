from abc import ABCMeta, abstractmethod
import os
from pathlib import Path
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
# from langchain_community.document_loaders.pdf import PDFMinerLoader
from langchain_community.document_loaders import UnstructuredXMLLoader, PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_experimental.text_splitter import SemanticChunker
from tqdm import tqdm
from itertools import groupby
import re

# Define a dictionary to map file extensions to their respective loaders
DEFAULT_LOADERS = {
    '.pdf': PyPDFLoader, # PDFMinerLoader
    '.xml': UnstructuredXMLLoader,
    '.csv': CSVLoader,
    '.txt': TextLoader,
    '.html': UnstructuredHTMLLoader,
}

def create_file_type_loader(file_type, directory_path):
    return DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_cls=DEFAULT_LOADERS[file_type],
    )


class BaseChunker(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def split(self, docs: list[Document]):
        pass
    

class RecursiveSplitter(BaseChunker):
    """
    Class to chunk/split documents into smaller sections.

    This uses a langchain TextSplitter to divide documents into
    smaller chunks of text.
    """

    def __init__(self, chunk_size: int=200, chunk_overlap: int=20, **args):
        """
        Initialize the Chunker object.

        Args:
            chunk_size: The maximum size of each chunk.
            chunk_overlap: The number of tokens of overlap between chunks.
        """


        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


    def split(self, docs: list[Document]) -> list[Document]:
        """Split a list of documents into chunks.

        Args:
            docs (list[Document]): List of documents to split.

        Returns:
            list[Document]: List of splitted documents
        """
        if not isinstance(docs, list):
            raise TypeError("docs must be a list of documents.")
        
        if len(docs) == 0:
            raise IndexError("Docs is empty.")
        
        s = self.text_splitter.split_documents(docs)
        return s    

class SemanticSplitter(BaseChunker):
    def __init__(self, min_chunk_size: int = 1000, embeddings = None, **args) -> None:
        self.text_splitter = SemanticChunker(
            embeddings=embeddings, 
            buffer_size=3, 
            breakpoint_threshold_type='percentile', 
            min_chunk_size=min_chunk_size
        )
    

    def split(self, docs: list[Document]):
        if not isinstance(docs, list):
            raise TypeError("docs must be a list of documents.")
        
        if len(docs) == 0:
            raise IndexError("Docs is empty.")
        
        return self.text_splitter.split_documents(docs)


class SemanticSplitter2(BaseChunker):
    def __init__(self, min_chunk_size: int = 900, embeddings = None, **args) -> None:
        self.text_splitter = SemanticChunker(
            embeddings=embeddings, 
            buffer_size=3, 
            breakpoint_threshold_type='percentile', 
            min_chunk_size=min_chunk_size,
            add_start_index=True
        )
    
    def split_doc(self, pages: list[Document], source: str):
        text = ''
        page_idx = []
        start_index = 0

        for page_num, p in enumerate(pages, start=1):
            text += ' ' + p.page_content
            c = ' '.join(
                re.split(self.text_splitter.sentence_split_regex, text)
            )
            end_index = len(c) -1
            page_idx.append(
                {"start_idx": start_index, "end_idx": end_index, "page": page_num}
            )
            start_index = end_index
        
        chunks = self.text_splitter.create_documents(
            [' '.join(re.split(self.text_splitter.sentence_split_regex, text))]
        )

        i = 0
        for semantic_chunk in chunks:
            start_idx = semantic_chunk.metadata["start_index"]
            while not (page_idx[i]["start_idx"] <= start_idx < page_idx[i]["end_idx"]):
                i += 1
            
            # print(start_idx, page_idx[i])
            semantic_chunk.metadata = {
                "page": page_idx[i]["page"], 
                "source": source
            }
        
        return chunks

    
    def split(self, docs: list[Document]):
        d = []
        docs.sort(key=lambda x: x.metadata["source"])
        for source, docs in groupby(docs, key=lambda x: x.metadata["source"]):
            d.extend(
                self.split_doc(docs, source)
            )
        return d



class Docs:
    """
    Class to represent and manage a collection of documents.

    This class handles:
        - Loading documents from a directory/file
        - Storing documents in a list
        - Splitting documents into chunks
        - Providing methods to operate on the documents

    Attributes:
        docs (list): The original full documents loaded from the directory.
        chunked_docs (list): The documents split into chunks.
    """

    def __init__(self, data_path: str =None):
        """
        Initialize the Documents object.

        Sets up empty lists to store the original and chunked documents.
        """
        self.data_path = data_path
        self.docs : list[Document] = []

    def load(self, dir_path: str | Path) -> None:
        """
        Load documents from the given directory.

        Loads documents of the following types:
            - PDF (.pdf)
            - XML (.xml)
            - CSV (.csv)
            - Plain text (.txt)
            - HTML (.html)

        Uses the appropriate loader for each document type.

        Args:
            dir_path (str): The path to the directory to load documents from.

        Returns:
            None

        Raises:
            FileNotFoundError: If dir_path does not exist.
            ValueError: If dir_path is invalid.
        """

        # Validate directory path
        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        for file_type in tqdm(DEFAULT_LOADERS.keys(), desc=f"Loading documents from '{dir_path}'"):
            loader: DirectoryLoader = create_file_type_loader(file_type=file_type, directory_path=dir_path) # create loader specific to that type
            docs = loader.load() # load document
            self.docs.extend(docs) # add them to the list


    def load_file(self, file_path: str | Path):
        file_path = Path(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix not in DEFAULT_LOADERS:
            raise TypeError(f"File type {file_path.suffix} not supported")
        
        # get loader
        file_loader = DEFAULT_LOADERS[file_path.suffix](file_path)
        docs = file_loader.load()
        self.docs.extend(docs)


    def get_docs(self) -> list:
        """
        Get the original full documents.

        Returns:
            list: The original documents loaded from the directory.
        """
        return self.docs


CHUNKERS = {
    "recursive_splitter": RecursiveSplitter,
    "semantic_splitter": SemanticSplitter,
    "semantic_splitter2": SemanticSplitter2
}

def get_chunker_cls(strategy_name: str) -> BaseChunker:
    # Retrieve the chunker class from the map and instantiate it
    chunker = CHUNKERS.get(strategy_name, None)
    if chunker is None:
        raise ValueError(f"Unknown chunking strategy: {strategy_name}")
    return chunker




