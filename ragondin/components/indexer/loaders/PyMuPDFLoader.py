from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader as pymupdf_loader
from langchain_core.documents.base import Document

from .base import BaseLoader


class PyMuPDFLoader(BaseLoader):
    def __init__(self, page_sep: str = "[PAGE_SEP]", **kwargs) -> None:
        super().__init__(**kwargs)
        self.page_sep = page_sep

    async def aload_document(self, file_path, metadata: dict = None, save_md=False):
        loader = pymupdf_loader(
            file_path=Path(file_path),
        )
        pages = await loader.aload()
        doc = Document(
            page_content=f"{self.page_sep}".join([p.page_content for p in pages]),
            metadata=metadata,
        )
        if save_md:
            self.save_document(doc, str(file_path))
        return doc
