from pathlib import Path

from langchain_core.documents.base import Document

from .base import BaseLoader
from typing import Optional

import json
import eml_parser


class EmlLoader(BaseLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def aload_document(self, file_path, metadata: dict = None, save_markdown: bool = False):
        path = Path(file_path)
        path = path.expanduser().resolve()

        with open(path, 'rb') as fhdl:
            raw_email = fhdl.read()

        ep = eml_parser.EmlParser(include_raw_body=True)
        parsed_eml = ep.decode_email_bytes(raw_email)
        content_body = parsed_eml['body'][0]['content']

        document = Document(page_content=content_body, metadata=metadata)

        if not document:
            raise ValueError(f"The file {file_path} could not be converted to Markdown.")
        
        return document
