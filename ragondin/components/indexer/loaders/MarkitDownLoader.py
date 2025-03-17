from components.indexer.loaders.BaseLoader import BaseLoader
from langchain_core.documents.base import Document
from components.utils import SingletonMeta
from langchain_openai import ChatOpenAI
from loguru import logger
import re
import base64
import os
from tqdm.asyncio import tqdm
from langchain_core.messages import HumanMessage
from langchain_core.documents.base import Document
import asyncio

class MarkItDownLoader(BaseLoader):
    """
    A loader class for converting documents to Markdown format and processing images within the documents.
    Attributes:
        page_sep (str): The separator used to denote page breaks in the document.
        converter (MarkItDownConverter): An instance of the MarkItDownConverter class.
        settings (dict): Configuration settings for the ChatOpenAI model.
        llm (ChatOpenAI): An instance of the ChatOpenAI class for language model interactions.
        semaphore (asyncio.Semaphore): A semaphore to limit the number of concurrent tasks.
    Methods:
        __init__(page_sep: str='[PAGE_SEP]', **kwargs) -> None:
            Initializes the MarkItDownLoader with the given page separator and additional keyword arguments.
        aload_document(file_path, metadata, save_md=False):
            Asynchronously loads and processes a document, optionally saving it in Markdown format.
        get_image_description(img_b64: str, image_ext, semaphore: asyncio.Semaphore):
            Asynchronously generates a description for a given image using the language model.
        get_captions(images):
            Asynchronously generates captions for a list of images.
        get_images_from_zip(input_file):
            Asynchronously extracts images from a ZIP file and returns them in the correct order.
        parse(file_path, page_seperator='[PAGE_SEP]'):
            Asynchronously parses a document, converts it to Markdown, and processes images within the document.
    """
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        super().__init__(**kwargs)
        from dotenv import load_dotenv
        
        self.page_sep = page_sep
        self.converter = MarkItDownConverter()
        self.settings = {
            'temperature': 0.2,
            'max_retries': 3,
            'timeout': 60,
            'api_key': api_key,
            'base_url': base_url,
            'model': "Qwen2.5-VL-7B-Instruct"
        }
        self.llm = ChatOpenAI(**self.settings)
        load_dotenv()
        api_key = os.getenv('API_KEY')
        base_url = os.getenv('BASE_URL')
        
        self.semaphore = asyncio.Semaphore(10)

    async def aload_document(self, file_path, metadata, save_md=False):
        result = await self.converter.convert_to_md(file_path)

        if self.config['loader']['image_captioning']:
            images = self.get_images_from_zip(file_path)
            captions = await self.get_captions(images)
            for caption in captions:
                result = re.sub(r"!\[[^!]*(\n){0,2}[^!]*\]\(data:image/.{3,4};base64...\)", caption,string=result, count=1)
        else:
            logger.info("Image captioning disabled. Ignoring images.")

        doc = Document(
            page_content=result,
            metadata=metadata
        )
        if save_md:
            self.save_document(Document(page_content=result), str(file_path))
        return doc

    async def get_image_description(self, img_b64: str, image_ext, semaphore: asyncio.Semaphore):
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        'url': f"data:image/{image_ext};base64,{img_b64}" #f"{picture.image.uri.path}" #
                    },
                },
                {
                    "type": "text", 
                    "text": """Provide a complete, structured and precise description of this image or figure in the same language (french) as its content. If the image contains tables, render them in markdown."""
                }
            ]
        )

        output = await self.llm.ainvoke([message])
        image_description = output.content
        markdown_content = (
            f"\nDescription de l'image:\n"
            f"{image_description}\n"
        )
        return markdown_content

    async def get_captions(self, images):
        tasks = [
            self.caption(
                img_b64=img,
                image_ext=image_ext,
                semaphore=self.semaphore
            )
            for img, image_ext in images
        ]
        return await tqdm.gather(*tasks, desc="Generating captions")
    
    async def get_images_from_zip(self, input_file):
        import zipfile
        with zipfile.ZipFile(input_file, 'r') as docx:
            file_names = docx.namelist()
            image_files = [f for f in file_names if f.startswith('word/media/')]

            images_not_in_order, order = [], []   # the images got from the original file is not in the right order
                                                # but the target_ref contains the position of the image in the document

            for image_file in image_files:
                image_data = docx.read(image_file)
                image_extension = image_file.split('.')[-1].lower()
                image = base64.b64encode(image_data).decode('utf-8')
                images_not_in_order.append((image, image_extension))
                order.append(image_file.split('media/image')[1].split(f'.{image_extension}')[0])
            
            images = [1] * len(images_not_in_order)   # the images in the right order
            for i in range(len(images_not_in_order)):
                images[int(order[i]) - 1] = images_not_in_order[i]
            return images

    async def parse(self, file_path, page_seperator='[PAGE_SEP]'):
        result = await self.converter.convert_to_md(file_path)
        
        images = self.get_images_from_zip(file_path)
        captions = await self.get_captions(images)
        for caption in captions:
            result = re.sub(r"!\[[^!]*(\n){0,2}[^!]*\]\(data:image/.{3,4};base64...\)", caption,string=result, count=1)
        return result

class MarkItDownConverter(metaclass=SingletonMeta):
    def __init__(self):
        try:
            from markitdown import MarkItDown
        except ImportError as e:
            logger.warning(f"MarkItDown is not installed. {e}. Install it using `pip install markitdown`")
            raise e
        
        self.converter = MarkItDown()
    
    async def converter(self, input_file: str):
        mdcontent = self.converter.convert(input_file).text_content
        return mdcontent
    
    async def convert_to_md(self, file_path):
        return await asyncio.to_thread(self.converter, str(file_path))

