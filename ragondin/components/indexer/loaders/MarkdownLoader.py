import re
from pathlib import Path
from langchain_core.documents.base import Document
from loguru import logger
from .base import BaseLoader


class MarkdownLoader(BaseLoader):
    def __init__(self, page_sep: str = "[PAGE_SEP]", **kwargs) -> None:
        super().__init__(**kwargs)
        self.page_sep = page_sep

        # Precompiled regex patterns for performance
        self._inline_img_pattern = re.compile(r"!\[.*?\]\(.*?\)")
        self._ref_img_pattern = re.compile(r"!\[.*?\]\[.*?\]")
        self._html_img_pattern = re.compile(r"<img[^>]*>")

    def _remove_images(self, text: str) -> str:
        """
        Remove Markdown and HTML image tags from the text.
        """
        text = self._inline_img_pattern.sub("", text)
        text = self._ref_img_pattern.sub("", text)
        text = self._html_img_pattern.sub("", text)
        return text

    async def aload_document(
        self, file_path: str, metadata: dict = None, save_md: bool = False
    ) -> Document:
        path = Path(file_path)
        raw_text = path.read_text(encoding="utf-8")

        # Remove all image references from the raw markdown
        clean_text = self._remove_images(raw_text)

        doc = Document(
            page_content=f"{self.page_sep}".join([clean_text]),
            metadata=metadata,
        )

        if save_md:
            self.save_document(doc, str(path))

        return doc
