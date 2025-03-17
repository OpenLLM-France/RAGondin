from pathlib import Path

from langchain_core.documents.base import Document
from langchain_community.document_loaders import TextLoader

from components.indexer.loaders.BaseLoader import BaseLoader

class CustomTextLoader(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        self.page_sep = page_sep

    async def aload_document(self, file_path, metadata: dict = None):
        path = Path(file_path)
        loader = TextLoader(file_path=str(path), autodetect_encoding=True)
        doc = await loader.aload()
        return Document(
            page_content=f'{self.page_sep}'.join([p.page_content for p in doc]), 
            metadata=metadata
        )