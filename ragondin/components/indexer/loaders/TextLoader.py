from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents.base import Document

from .base import BaseLoader


class CustomTextLoader(BaseLoader):
    def __init__(self, page_sep: str = "[PAGE_SEP]", **kwargs) -> None:
        super().__init__(**kwargs)
        self.page_sep = page_sep

    async def aload_document(self, file_path, metadata=None, save_md=False):
        path = Path(file_path)
        loader = TextLoader(file_path=str(path), autodetect_encoding=True)
        doc = await loader.aload()
        doc = Document(
            page_content=f"{self.page_sep}".join([p.page_content for p in doc]),
            metadata=metadata,
        )
        if save_md:
            self.save_document(doc, str(path))
        return doc
