from ragondin.components.indexer.loader import MarkItDownLoader
from ragondin.components.indexer.loaders import BaseLoader
from spire.doc import Document, FileFormat

import os

class MarkItDown_DocLoader(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        super().__init__(**kwargs)
        self.page_sep = page_sep
        self.MDLoader = MarkItDownLoader(page_sep=page_sep, **kwargs)

    async def aload_document(self, file_path, metadata , save_md=False):
        '''
        Here we convert the document to docx format, save it in local and then use the MarkItDownLoader 
        to convert it to markdown'''
        document = Document()
        document.LoadFromFile(file_path)
        file_path = "converted/sample.docx"
        document.SaveToFile(file_path, FileFormat.Docx2016)
        result_string = await self.MDLoader.aload_document(file_path, metadata, save_md)
        os.remove(file_path)
        document.Close()
        return result_string

    async def parse(self, file_path, page_seperator='[PAGE_SEP]'):
        document = Document()
        document.LoadFromFile(file_path)
        file_path = "converted/sample.docx"
        document.SaveToFile(file_path, FileFormat.Docx2016)
        result_string = await self.MDLoader.parse(file_path, page_seperator)
        os.remove(file_path)
        document.Close()
        return result_string
