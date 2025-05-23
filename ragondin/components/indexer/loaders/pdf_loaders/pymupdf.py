import asyncio
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader as pymupdf_loader
from langchain_core.documents.base import Document
import pymupdf4llm

from ..base import BaseLoader


class PyMuPDFLoader(BaseLoader):
    def __init__(self, page_sep="[PAGE_SEP]", **kwargs):
        super().__init__(page_sep, **kwargs)

    async def aload_document(
        self, file_path, metadata: dict = None, save_markdown=False
    ):
        loader = pymupdf_loader(
            file_path=Path(file_path),
        )
        pages = await loader.aload()
        doc = Document(
            page_content=f"{self.page_sep}".join([p.page_content for p in pages]),
            metadata=metadata,
        )
        if save_markdown:
            self.save_document(doc, str(file_path))
        return doc


class PyMuPDF4LLMLoader(BaseLoader):
    def __init__(self, page_sep: str = "[PAGE_SEP]", config=None, **kwargs) -> None:
        super().__init__(page_sep=page_sep, **kwargs)

    async def aload_document(
        self, file_path, metadata: dict = None, save_markdown=False
    ):
        pages = await asyncio.to_thread(
            pymupdf4llm.to_markdown,
            file_path,
            write_images=False,
            page_chunks=True,
        )

        page_content = f"{self.page_sep}".join([p["text"] for p in pages])
        doc = Document(page_content=page_content, metadata=metadata)
        if save_markdown:
            self.save_document(doc, str(file_path))
        return doc
