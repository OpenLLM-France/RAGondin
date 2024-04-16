from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.pdf import PDFMinerLoader
from langchain.document_loaders.xml import UnstructuredXMLLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders.merge import MergedDataLoader

# Define a dictionary to map file extensions to their respective loaders

DEFAULT_LOADERS = {
    '.pdf': PDFMinerLoader,
    '.xml': UnstructuredXMLLoader,
    '.csv': CSVLoader,
    '.txt': TextLoader,
    '.html': UnstructuredHTMLLoader,
}

class Documents:
    def __init__(self):
        self.docs : list[langchain_core.documents.base.Document] = []
        self.chuncked_docs : list[langchain_core.documents.base.Document] = []

    def load(self, dir_path: str)
    
        pdf_loader = create_directory_loader('.pdf', dir_path)
        xml_loader = create_directory_loader('.xml', dir_path)
        csv_loader = create_directory_loader('.csv', dir_path)
        txt_loader = create_directory_loader('.txt', dir_path)
        html_loader = create_directory_loader('.html', dir_path)

        #Load
        pdf_documents = pdf_loader.load()
        xml_documents = xml_loader.load()
        csv_documents = csv_loader.load()
        txt_documents = txt_loader.load()
        html_documents = html_loader.load()

        self.docs.append(pdf_documents + xml_documents + csv_documents + txt_documents + html_documents)

    def get_docs(self):
        return self.docs

    # Define a function to create a DirectoryLoader for a specific file type
    def create_directory_loader(file_type, directory_path):
        return DirectoryLoader(
            path=directory_path,
            glob=f"**/*{file_type}",
            loader_cls=DEFAULT_LOADERS[file_type],
        )

    def chunck(self, chuncker: Chuncker):
        self.chuncked_docs = chuncker.split(self.docs)
        return self.chuncked_docs

class Chuncker:
    def __init__(self, chunk_size: int: = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
    def split(self, docs: list[langchain_core.documents.base.Document]):
        self.text_splitter.split_documents(docs)
