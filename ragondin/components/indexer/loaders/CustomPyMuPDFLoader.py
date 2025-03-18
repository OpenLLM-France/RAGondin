from pathlib import Path

from langchain_core.documents.base import Document
from langchain_community.document_loaders import PyMuPDFLoader

from .base import BaseLoader

class CustomPyMuPDFLoader(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        self.page_sep = page_sep

    async def aload_document(self, file_path, metadata: dict = None):
        loader = PyMuPDFLoader(
            file_path=Path(file_path),
        )
        pages = await loader.aload()
        return Document(
            page_content=f'{self.page_sep}'.join([p.page_content for p in pages]), 
            metadata=metadata
        )