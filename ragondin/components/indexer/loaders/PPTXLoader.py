import tqdm
import re

from ragondin.components.indexer.loaders import BaseLoader
from ragondin.components.indexer.utils.pptx_converter import PPTXConverter

from langchain_core.documents.base import Document


class PPTXLoader(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        super().__init__(**kwargs)
        self.page_sep = page_sep
        self.image_placeholder=r"<image>"

        self.converter = PPTXConverter(
            image_placeholder=self.image_placeholder,
            page_separator=page_sep
        )
    
    async def get_captions(self, images):
        tasks = [
            self.get_image_description(image=img)
            for img in images
        ]
        return await tqdm.gather(*tasks, desc="Generating captions")


    async def aload_document(self, file_path, metadata, save_md):
        md_content, imgs = self.converter.convert(local_path=file_path)
        images_captions = await self.get_captions(imgs)

        for caption in images_captions:
            md_content = re.sub(self.image_placeholder, caption, md_content, count = 1)

        doc = Document(
            page_content=md_content,
            metadata=metadata
        )
        if save_md:
            self.save_document(Document(page_content=md_content), str(file_path))
        return doc

   