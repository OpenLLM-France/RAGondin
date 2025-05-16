from .base import BaseLoader
from PIL import Image
from pathlib import Path
from langchain_core.documents import Document


class ImageLoader(BaseLoader):
    def __init__(self, page_sep="[PAGE_SEP]", **kwargs):
        super().__init__(page_sep, **kwargs)

    async def aload_document(self, file_path, metadata=None, save_md=False):
        path = Path(file_path)
        img = Image.open(path)
        description = await self.get_image_description(image=img)
        doc = Document(page_content=description, metadata=metadata)
        if save_md:
            self.save_document(doc, str(path))

        return doc
