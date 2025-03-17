from pathlib import Path

from langchain_core.documents.base import Document
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, UnstructuredODTLoader

from components.indexer.loaders.BaseLoader import BaseLoader


class CustomDocLoader(BaseLoader):
    """
    Custom document loader that supports asynchronous loading of various document formats.
    Attributes:
        doc_loaders (dict): A dictionary mapping file extensions to their respective loader classes.
        page_sep (str): A string used to separate pages in the loaded document.
    Methods:
        __init__(page_sep: str='[PAGE_SEP]', **kwargs) -> None:
            Initializes the CustomDocLoader with an optional page separator.
        aload_document(file_path: str, metadata: dict = None) -> Document:
            Asynchronously loads a document from the given file path and returns a Document object.
            Raises a ValueError if the file format is not supported.
    """
    doc_loaders = {
            ".docx": UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.odt': UnstructuredODTLoader
        }
    
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        self.page_sep = page_sep
    
    
    async def aload_document(self, file_path, metadata: dict = None):
        path = Path(file_path)
        cls_loader = CustomDocLoader.doc_loaders.get(path.suffix, None)

        if cls_loader is None:
            raise ValueError(f"This loader only supports {CustomDocLoader.doc_loaders.keys()} format")
        
        loader = cls_loader(
            file_path=str(file_path), 
            mode='single',
        )
        pages = await loader.aload()
        content = f'{self.page_sep}'.join([p.page_content for p in pages])

        return Document(
            page_content=f"{content}{self.page_sep}", 
            metadata=metadata
        )