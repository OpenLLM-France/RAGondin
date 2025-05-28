"""
Text file loader implementation.
"""

from pathlib import Path
from typing import Dict, Optional, Union
from langchain_community.document_loaders import TextLoader as LangchainTextLoader
from langchain_core.documents.base import Document
from components.indexer.loaders.base import BaseLoader


class TextLoader(BaseLoader):
    """
    Loader for plain text files (.txt).
    """

    def __init__(self, page_sep: str = "[PAGE_SEP]", **kwargs) -> None:
        super().__init__(page_sep=page_sep, **kwargs)

    async def aload_document(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict] = None,
        save_markdown: bool = False,
    ) -> Document:
        if metadata is None:
            metadata = {}

        path = Path(file_path)
        loader = LangchainTextLoader(file_path=str(path), autodetect_encoding=True)

        # Load document segments asynchronously
        doc_segments = await loader.aload()

        # Create final document
        doc = Document(
            page_content=f"{self.page_sep}".join(
                [s.page_content for s in doc_segments]
            ),
            metadata=metadata,
        )

        # Save if requested
        if save_markdown:
            self.save_document(doc, str(path))

        return doc
