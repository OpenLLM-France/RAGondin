import os

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

def create_directory_loader(file_type, directory_path):
    return DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_cls=DEFAULT_LOADERS[file_type],
    )

class Chunker:
    """
    Class to chunk/split documents into smaller sections.

    This uses a langchain TextSplitter to divide documents into
    smaller chunks of text.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the Chunker object.

        Args:
            chunk_size: The maximum size of each chunk.
            chunk_overlap: The number of tokens of overlap between chunks.
        """

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(self, docs: list[Document]):
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
            raise ValueError("docs must be a list")

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

    def __init__(self):
        """
        Initialize the Documents object.

        Sets up empty lists to store the original and chunked documents.
        """

        self.docs : list[Document] = []
        self.chunked_docs : list[Document] = []

    def load(self, dir_path) -> None:
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

        # Create loader for each file type
        pdf_loader = create_directory_loader('.pdf', dir_path)
        xml_loader = create_directory_loader('.xml', dir_path)
        csv_loader = create_directory_loader('.csv', dir_path)
        txt_loader = create_directory_loader('.txt', dir_path)
        html_loader = create_directory_loader('.html', dir_path)

        # Load documents with each loader
        pdf_docs = pdf_loader.load()
        xml_docs = xml_loader.load()
        csv_docs = csv_loader.load()
        txt_docs = txt_loader.load()
        html_docs = html_loader.load()

        # Add all docs to main list
        self.docs.extend(pdf_docs)
        self.docs.extend(xml_docs)
        self.docs.extend(csv_docs)
        self.docs.extend(txt_docs)
        self.docs.extend(html_docs)

    def chunk(self, chunker: Chunker) -> None:
        """
        Split loaded documents into chunks.

        Uses the provided Chunker instance to split the loaded
        documents into smaller chunks.

        Args:
            chunker (Chunker): The Chunker instance to use for splitting.

        Returns:
            None

        Raises:
            ValueError: If no documents are loaded.
        """

        if len(self.docs) == 0:
            raise ValueError("No documents loaded")

        # Perform splitting with Chunker
        # Store chunked docs
        self.chunked_docs = chunker.split(self.docs)

    def get_docs(self) -> list:
        """
        Get the original full documents.

        Returns:
            list: The original documents loaded from the directory.
        """
        return self.docs

    def get_chunks(self) -> list:
        """
        Get the chunked documents.

        Returns:
            list: The documents split into chunks.
        """
        return self.chunked_docs

