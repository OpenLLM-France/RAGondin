from langchain_core.documents.base import Document
from .base import BaseLoader
from PIL import Image

class ImageLoader(BaseLoader):
    def __init__(self, page_sep = "[PAGE_SEP]", **kwargs):
        super().__init__(**kwargs)

        self.page_sep = page_sep

    async def aload_document(self, file_path, metadata = None, save_md=False):
        image = Image.open(file_path)  # str file_path -> PIL.Image object
        result = await self.get_image_description(image=image)
        doc = Document(page_content=result, metadata=metadata)
        if save_md:
            self.save_document(Document(page_content=result), str(file_path))
        return doc

