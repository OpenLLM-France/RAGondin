import pymupdf4llm

from langchain_core.documents.base import Document
from components.indexer.loaders.BaseLoader import BaseLoader

class Custompymupdf4llm(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', config=None, **kwargs) -> None:
        self.page_sep = page_sep
    
    async def aload_document(self, file_path, metadata: dict = None):
        pages = pymupdf4llm.to_markdown(
            file_path,
            write_images=False,
            page_chunks=True,
        )
        page_content = f'{self.page_sep}'.join([p['text'] for p in pages])
        return Document(
            page_content=page_content, 
            metadata=metadata
        )