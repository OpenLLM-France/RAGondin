import asyncio
import pymupdf4llm
from langchain_core.documents.base import Document

from ..base import BaseLoader


class PyMuPDF4LLMLoader(BaseLoader):
    def __init__(self, page_sep: str = "[PAGE_SEP]", config=None, **kwargs) -> None:
        super().__init__(page_sep=page_sep, **kwargs)

    async def aload_document(self, file_path, metadata: dict = None, save_md=False):
        pages = await asyncio.to_thread(
            pymupdf4llm.to_markdown,
            file_path,
            write_images=False,
            page_chunks=True,
        )

        page_content = f"{self.page_sep}".join([p["text"] for p in pages])
        doc = Document(page_content=page_content, metadata=metadata)
        if save_md:
            self.save_document(doc, str(file_path))
        return doc
