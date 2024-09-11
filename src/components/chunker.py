from abc import ABCMeta, abstractmethod
import os
from pathlib import Path
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.pdf import PDFMinerLoader
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader

# Define a dictionary to map file extensions to their respective loaders

DEFAULT_LOADERS = {
    '.pdf': PDFMinerLoader,
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
    def __init__(self, 
            chunk_size: int=200, 
            chunk_overlap: int=20, 
            chunker_args: dict=None
        ) -> None:
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

    def __init__(self, chunk_size: int=200, chunk_overlap: int=20, chunker_args: dict=None):
        """
        Initialize the Chunker object.

        Args:
            chunk_size: The maximum size of each chunk.
            chunk_overlap: The number of tokens of overlap between chunks.
        """

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **chunker_args
        )

    def split(self, docs: list[Document]=[]):
        """
        Split a list of documents into chunks.

        Uses the langchain TextSplitter to divide the text
        from the documents into smaller chunks.

        Args:
            docs: List of documents to split.

        Returns:
            None

        Raises:
            ValueError: If docs is not a list.
        """

        if not isinstance(docs, list):
            raise TypeError("docs must be a list of documents.")
        
        if len(docs) == 0:
            raise IndexError("Docs is empty.")

        return self.text_splitter.split_documents(docs)


class Docs:
    """
    Class to represent and manage a collection of documents.

    This class handles:
        - Loading documents from a directory
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
        # self.chunked_docs : list[Document] = []
        # self.load(dir_path=data_path)

    def __getitem__(self, item):
        return self.chunked_docs[item]

    def __len__(self):
        return len(self.chunked_docs)

    def __iter__(self):
        return iter(self.chunked_docs)


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
        
        for file_type in DEFAULT_LOADERS.keys():
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
        
        


    # def chunk(self, chunker: RecursiveSplitter) -> None:
    #     """
    #     Split loaded documents into chunks.

    #     Uses the provided Chunker instance to split the loaded
    #     documents into smaller chunks.

    #     Args:
    #         chunker (Chunker): The Chunker instance to use for splitting.

    #     Returns:
    #         None

    #     Raises:
    #         ValueError: If no documents are loaded.
    #     """

    #     if len(self.docs) == 0:
    #         raise ValueError("No documents loaded")

    #     # Perform splitting with Chunker
    #     # Store chunked docs
    #     self.chunked_docs = chunker.split(self.docs)


    def get_docs(self) -> list:
        """
        Get the original full documents.

        Returns:
            list: The original documents loaded from the directory.
        """
        return self.docs

    # def get_chunks(self) -> list[Document]:
    #     """
    #     Get the chunked documents.

    #     Returns:
    #         list: The documents split into chunks.
    #     """
    #     return self.chunked_docs


CHUNKERS = {
    "recursive_splitter": RecursiveSplitter
}

def get_chunker_cls(strategy_name: str) -> BaseChunker:
    # Retrieve the chunker class from the map and instantiate it
    chunker = CHUNKERS.get(strategy_name, None)
    if chunker is None:
        raise ValueError(f"Unknown chunking strategy: {strategy_name}")
    return chunker




