from abc import abstractmethod, ABC
import asyncio
import html
import os
from collections import defaultdict
import gc
from langchain_openai import ChatOpenAI
from openai import OpenAI
import pptx
from components.utils import SingletonMeta
from pydub import AudioSegment
import torch
from pathlib import Path
from typing import AsyncGenerator, Optional, Dict, Union
from langchain_core.documents.base import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    UnstructuredODTLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.messages import HumanMessage

import pymupdf4llm
from loguru import logger
from aiopath import AsyncPath
from typing import Dict
from docling_core.types.doc.document import PictureItem
from docling.datamodel.document import ConversionResult

from tqdm.asyncio import tqdm
from ..utils import llmSemaphore, SingletonABCMeta
import re
import base64
from io import BytesIO
from PIL import Image
from markitdown import MarkItDown
from components.indexer.utils.pptx_converter import PPTXConverter

import time


# from langchain_community.document_loaders import UnstructuredXMLLoader, PyPDFLoader
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.document_loaders.text import TextLoader
# from langchain_community.document_loaders import UnstructuredHTMLLoader

class BaseLoader(ABC):
    def __init__(self, **kwargs) -> None:
        self.config = kwargs.get('config')
        llm_config = self.config["llm"]
        model_settings = {
            'temperature': 0.2,
            'max_retries': 3,
            'timeout': 60,
        }
        settings: dict = llm_config
        settings.update(model_settings)

        self.vlm_endpoint = ChatOpenAI(**settings).with_retry(stop_after_attempt=2)
        self.min_width_pixels = 100  # minimum width in pixels
        self.min_height_pixels = 100  # minimum height in pixels

    @abstractmethod
    async def aload_document(self, file_path, metadata: dict=None, save_md=False):
        pass

    def save_document(self, doc: Document, path: str):
        path = re.sub(r'\..*', '.md', path)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(doc.page_content)

    async def get_image_description(self, image, semaphore:asyncio.Semaphore=llmSemaphore):
        """
        Creates a description for an image using the LLM model defined in the constructor
        Args:
            image (PIL.Image): Image to describe
            semaphore (asyncio.Semaphore): Semaphore to control access to the LLM model
            Returns:
            str: Description of the image
        """
        async with semaphore:
            width, height = image.size

            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            image_description = ""

            message = HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            'url': f"data:image/png;base64,{img_b64}" #f"{picture.image.uri.path}" #
                        },
                    },
                    {
                        "type": "text", 
                        "text": """Provide a complete, structured and precise description of this image or figure in the same language (french) as its content. If the image contains tables, render them in markdown."""
                    }
                ]
            )
            try:
                if (width > self.min_width_pixels and height > self.min_height_pixels):
                    response = await self.vlm_endpoint.ainvoke([message])
                    image_description = response.content

            except Exception as e:
                logger.error(f"Error while generating image description: {e}")

            # Convert image path to markdown format and combine with description
            if image_description:
                markdown_content = (
                    f"\nDescription de l'image:\n"
                    f"{image_description}\n"
                )
            else:
                markdown_content = ''
                
            return markdown_content


class Custompymupdf4llm(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', config=None, **kwargs) -> None:
        self.page_sep = page_sep
    
    async def aload_document(self, file_path, metadata: dict = None):
        pages = pymupdf4llm.to_markdown(
            file_path,
            write_images=False,
            page_chunks=True,
        )
        page_content = f'{self.page_sep}'.join([p['text'] for p in pages])
        return Document(
            page_content=page_content, 
            metadata=metadata
        )


class AudioTranscriber(metaclass=SingletonMeta):
    def __init__(self, device='cpu', compute_type='float32', model_name='large-v2', language='fr'):
        self.model = whisperx.load_model(
            model_name, 
            device=device, language=language, 
            compute_type=compute_type
        )
        
class VideoAudioLoader(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', batch_size=4, config=None):    
        self.batch_size = batch_size
        self.page_sep = page_sep
        self.formats = [".wav", '.mp3', ".mp4"]

        self.transcriber = AudioTranscriber(
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    def free_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def aload_document(self, file_path, metadata: dict = None):
        path = Path(file_path)
        if path.suffix not in self.formats:
            logger.warning(
                f'This audio/video file ({path.suffix}) is not supported.'
                f'The format should be {self.formats}'
            )
            return None

        # load it in wave format
        if path.suffix == '.wav':
            audio_path_wav = path
        else:
            sound = AudioSegment.from_file(file=path, format=path.suffix[1:])
            audio_path_wav = path.with_suffix('.wav')
            # Export to wav format
            sound.export(
                audio_path_wav, format="wav", 
                parameters=["-ar", "16000", "-ac", "1", "-ab", "32k"]
            )
        
        audio = whisperx.load_audio(audio_path_wav)

        if path.suffix != '.wav':
            os.remove(audio_path_wav)

        transcription_l = self.transcriber.model.transcribe(audio, batch_size=self.batch_size)
        content = ' '.join([tr['text'] for tr in transcription_l['segments']])

        self.free_memory()

        return Document(
            page_content=f"{content}{self.page_sep}",
            metadata=metadata
        )


class CustomPPTLoader(BaseLoader):
    doc_loaders = {
        '.pptx': UnstructuredPowerPointLoader, 
        '.ppt': UnstructuredPowerPointLoader
    }

    cat2md = {
        'Title': '#',
        'NarrativeText': '>',
        'ListItem': '-',
        'UncategorizedText': '',
        'EmailAddress': '',
        'FigureCaption': '*'
    }
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        self.loader_args = kwargs
        self.page_sep = page_sep
    
    def group_by_hierarchy(self, docs: list[Document]):
        """Group related elements within each page by parent_id and category_depth"""
        grouped_elements = defaultdict(list)
        
        # Sort by depth to ensure proper hierarchy
        sorted_docs = sorted(docs, key=lambda x: x.metadata.get('category_depth', 0))
        
        for doc in sorted_docs:
            parent_id = doc.metadata.get('parent_id', None)
            if parent_id:
                grouped_elements[parent_id].append(doc)
            else:
                grouped_elements[doc.metadata.get('element_id')].append(doc)
        return grouped_elements
    
    async def aload_document(self, file_path, metadata: dict = None):
        path = Path(file_path)
        cls_loader = CustomPPTLoader.doc_loaders.get(path.suffix, None)

        if cls_loader is None:
            raise Exception(f"This loader only supports {CustomPPTLoader.doc_loaders.keys()} format")
        
        loader = cls_loader(
            file_path=str(file_path), 
            mode='elements',
            **self.loader_args
        )
        elements = await loader.aload()

        # Step 1: Group elements by page_number
        grouped_by_page = defaultdict(list)
        for doc in elements:
            page_number = doc.metadata.get('page_number')
            grouped_by_page[page_number].append(doc)
        
        # Final structure: dictionary of pages, each containing grouped elements
        final_grouped_structure = {}
        for page, docs in grouped_by_page.items():
            final_grouped_structure[page] = self.group_by_hierarchy(docs)

        content = ''
        for page, grouped_elements in final_grouped_structure.items():
            t = ''
            for v in grouped_elements.values():
                for elem in v:
                    meta = elem.metadata
                    cat_md = CustomPPTLoader.cat2md.get(meta.get("category"), '')
                    t += f"{cat_md} {elem.page_content}\n"

            if page is not None and content:
                t = f'{self.page_sep}' + t
            
            content += t

        return Document(
            page_content=content, 
            metadata=metadata
        )

class CustomPyMuPDFLoader(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        self.page_sep = page_sep

    async def aload_document(self, file_path, metadata: dict = None):
        loader = PyMuPDFLoader(
            file_path=Path(file_path),
        )
        pages = await loader.aload()
        return Document(
            page_content=f'{self.page_sep}'.join([p.page_content for p in pages]), 
            metadata=metadata
        )

class CustomTextLoader(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        self.page_sep = page_sep

    async def aload_document(self, file_path, metadata: dict = None):
        path = Path(file_path)
        loader = TextLoader(file_path=str(path), autodetect_encoding=True)
        doc = await loader.aload()
        return Document(
            page_content=f'{self.page_sep}'.join([p.page_content for p in doc]), 
            metadata=metadata
        )
    

class CustomHTMLLoader(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        self.page_sep = page_sep

    async def aload_document(self, file_path, metadata: dict = None):
        path = Path(file_path)
        loader = UnstructuredHTMLLoader(file_path=str(path), autodetect_encoding=True)
        doc = await loader.aload()
        return Document(
            page_content=f'{self.page_sep}'.join([p.page_content for p in doc]), 
            metadata=metadata
        )


class CustomDocLoader(BaseLoader):
    """
    Custom document loader that supports asynchronous loading of various document formats.
    Attributes:
        doc_loaders (dict): A dictionary mapping file extensions to their respective loader classes.
        page_sep (str): A string used to separate pages in the loaded document.
    Methods:
        __init__(page_sep: str='[PAGE_SEP]', **kwargs) -> None:
            Initializes the CustomDocLoader with an optional page separator.
        aload_document(file_path: str, metadata: dict = None) -> Document:
            Asynchronously loads a document from the given file path and returns a Document object.
            Raises a ValueError if the file format is not supported.
    """
    doc_loaders = {
            ".docx": UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.odt': UnstructuredODTLoader
        }
    
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        self.page_sep = page_sep
    
    
    async def aload_document(self, file_path, metadata: dict = None):
        path = Path(file_path)
        cls_loader = CustomDocLoader.doc_loaders.get(path.suffix, None)

        if cls_loader is None:
            raise ValueError(f"This loader only supports {CustomDocLoader.doc_loaders.keys()} format")
        
        loader = cls_loader(
            file_path=str(file_path), 
            mode='single',
        )
        pages = await loader.aload()
        content = f'{self.page_sep}'.join([p.page_content for p in pages])

        return Document(
            page_content=f"{content}{self.page_sep}", 
            metadata=metadata
        )
    

class DoclingConverter(metaclass=SingletonMeta):
    """
    A class to handle document conversion using the Docling library.
    Attributes:
    -----------
    converter : DocumentConverter
        An instance of the DocumentConverter class from the Docling library configured with specific pipeline options.
    Methods:
    --------
    __init__():
        Initializes the DoclingConverter with specific pipeline options for PDF conversion.
    async convert_to_md(file_path) -> ConversionResult:
        Asynchronously converts a document at the given file path to Markdown format.
    """
    def __init__(self):
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.document import ConversionResult
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                AcceleratorDevice,
                AcceleratorOptions,
                PdfPipelineOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
            from docling.datamodel.pipeline_options import (
                TableStructureOptions,
                TableFormerMode
            )

        except ImportError as e:
            logger.warning(f"Docling is not installed. {e}. Install it using `pip install docling`")
            raise e
        
        img_scale = 2
        pipeline_options = PdfPipelineOptions(
            do_ocr = True,
            do_table_structure = True,
            generate_picture_images=True,
            images_scale=img_scale
            # generate_table_images=True,
            # generate_page_images=True
        )
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True,
            mode=TableFormerMode.ACCURATE
        )

        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=12, device=AcceleratorDevice.AUTO
        )
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
                )
            }
        )
    
    async def convert_to_md(self, file_path) -> ConversionResult:
        return await asyncio.to_thread(self.converter.convert, str(file_path))

class DoclingLoader(BaseLoader):
    """
    DoclingLoader is responsible for loading and processing documents, converting them to markdown format,
    and optionally enriching them with image captions.
    Attributes:
        page_sep (str): The separator used between pages in the markdown output.
        converter (DoclingConverter): An instance of the DoclingConverter class used for document conversion.
    Methods:
        __init__(page_sep: str='[PAGE_SEP]', **kwargs) -> None:
            Initializes the DoclingLoader with the given page separator and additional keyword arguments.
        async aload_document(file_path, metadata, save_md=False):
            Asynchronously loads and processes a document, converting it to markdown and optionally saving it.
            Args:
                file_path (str): The path to the document file.
                metadata (dict): Metadata associated with the document.
                save_md (bool): Whether to save the markdown content to a file.
        async get_captions(pictures: list[PictureItem], n_semaphores=10):
            Asynchronously retrieves captions for a list of pictures using a specified number of semaphores.
            Args:
                pictures (list[PictureItem]): A list of PictureItem objects to caption.
                n_semaphores (int): The number of semaphores to use for concurrent captioning.
        async convert_to_md(file_path) -> ConversionResult:
            Asynchronously converts a document to markdown format.
            Args:
                file_path (str): The path to the document file.
        async parse(file_path, page_seperator='[PAGE_SEP]'):
            Asynchronously parses a document, converting it to markdown and enriching it with image captions.
            Args:
                file_path (str): The path to the document file.
                page_seperator (str): The separator used between pages in the markdown output.
    """
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        super().__init__(**kwargs)
        self.page_sep = page_sep
        self.converter = DoclingConverter()
    
    async def aload_document(self, file_path, metadata, save_md=False):
        result = await self.converter.convert_to_md(file_path)


        n_pages = len(result.pages)
        s = f'{self.page_sep}'.join([result.document.export_to_markdown(page_no=i) for i in range(1, n_pages+1)])

        enriched_content = s

        if self.config["loader"]["image_captioning"]:
            pictures = result.document.pictures
            descriptions = await self.get_captions(pictures)
            for description in descriptions:
                enriched_content = enriched_content.replace('<!-- image -->', description, 1)
        else:
            logger.info("Image captioning disabled. Ignoring images.")

        doc =  Document(
            page_content=enriched_content, 
            metadata=metadata
        )
        if save_md:
            self.save_document(Document(page_content=enriched_content), str(file_path))
        return doc
        
    
    async def get_captions(self, pictures: list[PictureItem]):
        tasks = []

        for picture in pictures:
            tasks.append(
                self.get_image_description(picture.image.pil_image)
            )
        try:
            results = await tqdm.gather(*tasks, desc='Captioning imgs')  # asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            
            raise
        return results
    
    async def convert_to_md(self, file_path) -> ConversionResult:
        return await asyncio.to_thread(self.converter.convert, str(file_path))
    
    async def parse(self, file_path, page_seperator='[PAGE_SEP]'):
        result = await self.convert_to_md(file_path)
        n_pages = len(result.pages)
        s = f'{page_seperator}'.join([result.document.export_to_markdown(page_no=i) for i in range(1, n_pages+1)])
        pictures = result.document.pictures
        descriptions = await self.get_captions(pictures)
        enriched_content = s
        for description in descriptions:
            enriched_content = enriched_content.replace('<!-- image -->', description, 1)
        return enriched_content
    
class MarkItDownLoader(BaseLoader):
    """
    A loader class for converting documents to Markdown format and processing images within the documents.
    Attributes:
        page_sep (str): The separator used to denote page breaks in the document.
        converter (MarkItDownConverter): An instance of the MarkItDownConverter class.
        settings (dict): Configuration settings for the ChatOpenAI model.
        llm (ChatOpenAI): An instance of the ChatOpenAI class for language model interactions.
        semaphore (asyncio.Semaphore): A semaphore to limit the number of concurrent tasks.
    """
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        super().__init__(**kwargs)
       
        self.page_sep = page_sep
        self.converter = MarkItDown()

    async def aload_document(self, file_path, metadata, save_md=False):
        result = self.converter.convert(file_path).text_content

        if self.config['loader']['image_captioning']:
            images = self.get_images_from_zip(file_path)
            captions = await self.get_captions(images)
            for caption in captions:
                result = re.sub(r"!\[[^!]*(\n){0,2}[^!]*\]\(data:image/.{0-6};base64...\)", caption.replace("\\","/"), string=result, count=1)
        else:
            logger.info("Image captioning disabled. Ignoring images.")

        doc = Document(
            page_content=result,
            metadata=metadata
        )
        if save_md:
            self.save_document(Document(page_content=result), str(file_path))
        return doc

    async def get_captions(self, images):
        tasks = [
            self.get_image_description(image=img)
            for img, image_ext in images
        ]
        return await tqdm.gather(*tasks, desc="Generating captions")
    
    def get_images_from_zip(self, input_file):
        import zipfile
        with zipfile.ZipFile(input_file, 'r') as docx:
            file_names = docx.namelist()
            image_files = [f for f in file_names if f.startswith('word/media/')]

            images_not_in_order, order = [], []   # the images got from the original file is not in the right order
                                                # but the target_ref contains the position of the image in the document

            for image_file in image_files:
                image_data = docx.read(image_file)
                image_extension = image_file.split('.')[-1].lower()
                image = Image.open(BytesIO(image_data))
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
            result = re.sub(r"!\[[^!]*(\n){0,2}[^!]*\]\(data:image/.{0-6};base64...\)", caption.replace("\\","/"),string=result, count=1)
        return result

class MarkItDown_DocLoader(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        super().__init__(**kwargs)
        from spire.doc import Document, FileFormat
        from spire.doc.common import FileFormat
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

   

class MarkerConverter(metaclass=SingletonMeta):
    """
    A class used to convert files to markdown format using a PDF converter.
    Attributes
    ----------
    converter : PdfConverter
        An instance of PdfConverter initialized with specific configuration.
    Methods
    -------
    convert_to_md(file_path)
        Asynchronously converts the given file to markdown format.
    """
    def __init__(self) -> None:

        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        self.converter = PdfConverter(
            artifact_dict=create_model_dict(),
            config={
                    'output_format': 'markdown',
                    'paginate_output': True,
                }
        )
    
    async def convert_to_md(self, file_path):
        return await asyncio.to_thread(self.converter, str(file_path))

class MarkerLoader(BaseLoader):
    """
    MarkerLoader is a class responsible for loading and converting documents into markdown format,
    with optional image captioning using a language model.
    Attributes:
        page_sep (str): Separator used for pages in the document.
        converter (MarkerConverter): Instance of MarkerConverter for converting documents to markdown.
    Methods:
        __init__(page_sep: str='------------------------------------------------\n\n', **kwargs) -> None:
            Initializes the MarkerLoader with the given page separator and additional keyword arguments.
        async aload_document(file_path, metadata=None, save_md=False):
            Asynchronously loads a document from the specified file path, converts it to markdown,
            optionally replaces image tags with descriptions, and returns a Document object.
        async get_captions(img_dict, n_semaphores=10):
            Asynchronously generates captions for images in the document using a language model,
            with a specified number of semaphores for concurrency control.
    """
    def __init__(self, page_sep: str='------------------------------------------------\n\n', **kwargs) -> None:
        super().__init__(**kwargs)

        self.page_sep = page_sep
        self.converter = MarkerConverter()

    async def aload_document(self, file_path, metadata=None, save_md=False):
        file_path = str(file_path)
        logger.info(f"Loading {file_path}")
        start = time.time()
        render = await self.converter.convert_to_md(file_path)
        conversion_time = time.time() - start
        logger.info(f"Markdown conversion time: {conversion_time:.2f} s.")

        text = render.markdown

        # Parse and replace images with llm based descriptions
        # Find all instances of markdown image tags
        if self.config["loader"]["image_captioning"]:
            img_dict = render.images
            logger.info(f"Found {len(img_dict)} images in the document.")

            captions_dict = await self.get_captions(img_dict)

            for key, desc in captions_dict.items():
                tag = f'![]({key})'
                text = text.replace(tag, desc)
        else:
            logger.info("Image captioning disabled. Ignoring images.")

        end = time.time()
        logger.info(f"Total conversion time for file {file_path}: {end - start:.2f} s.")

        doc =  Document(
            page_content=text, 
            metadata=metadata
        )

        if save_md:
            self.save_document(Document(page_content=text), str(file_path))

        return doc
    
    async def get_captions(self, img_dict):
        tasks = []

        for _, picture in img_dict.items():
            tasks.append(
                self.get_image_description(picture)
            )
        try:
            results = await tqdm.gather(*tasks, desc='Captioning imgs')  # asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            
            raise

        result_dict = dict(zip(img_dict.keys(), results))
        return result_dict


class DocSerializer:
    """
    A class used to serialize documents asynchronously.
    Attributes:
        data_dir (str, optional): The directory where the data is stored.
        kwargs (dict): Additional keyword arguments to pass to the loader.
    Methods:
        serialize_document(path: str, semaphore: asyncio.Semaphore, metadata: Optional[Dict] = {}) -> Document:
            Asynchronously serializes a single document from the given path.
        serialize_documents(path: str | Path | list[str], metadata: Optional[Dict] = {}, recursive=True, n_concurrent_ops=3) -> AsyncGenerator[Document, None]:
    """
    def __init__(self, data_dir=None, **kwargs) -> None:
        self.data_dir = data_dir
        self.kwargs = kwargs
        self.config = kwargs.get('config')

        self.loader_classes = get_loaders(self.config)
    
    async def serialize_document(self, path: str, semaphore: asyncio.Semaphore, metadata: Optional[Dict] = {}):
        p = AsyncPath(path)
        type_ = p.suffix
        loader_cls: BaseLoader = self.loader_classes.get(type_)
        sub_url_path = Path(path).resolve().relative_to(self.data_dir) # for the static file server
    
        if type_ == '.pdf': # for loaders that uses gpu
            async with semaphore:
                logger.debug(f'LOADING: {p.name}')
                loader = loader_cls(**self.kwargs)
                metadata={
                    'source': str(path),
                    'file_name': p.name,
                    'sub_url_path': str(sub_url_path),
                    'page_sep': loader.page_sep,
                    **metadata
                }
                doc: Document = await loader.aload_document(
                    file_path=path,
                    metadata=metadata,
                    save_md=True
                )
        else:
            logger.debug(f'LOADING: {p.name}')
            loader = loader_cls(**self.kwargs)  # Propagate kwargs here!
            metadata={
                'source': str(path),
                'file_name': p.name,
                'sub_url_path': str(sub_url_path),
                'page_sep': loader.page_sep,
                **metadata
            }
            doc: Document = await loader.aload_document(
                file_path=path,
                metadata=metadata,
                save_md=True
            )

        logger.info(f"{p.name}: SERIALIZED")
        return doc

    async def serialize_documents(self, path: str | Path | list[str], metadata: Optional[Dict] = {}, recursive=True, n_concurrent_ops=3) -> AsyncGenerator[Document, None]:
        """
        Asynchronously serializes documents from the given path(s).
        Args:
            path (str | Path | list[str]): The path or list of paths to the documents to be serialized.
            metadata (Optional[Dict], optional): Additional metadata to include with each document. Defaults to {}.
            recursive (bool, optional): Whether to search for files recursively in the given path(s). Defaults to True.
            n_concurrent_ops (int, optional): The number of concurrent operations to allow. Defaults to 3.
        Yields:
            AsyncGenerator[Document, None]: An asynchronous generator that yields serialized Document objects.
        """
        semaphore = asyncio.Semaphore(n_concurrent_ops)
        tasks = []
        async for file in get_files(self.loader_classes, path, recursive):
            tasks.append(
                self.serialize_document(
                    file,
                    semaphore=semaphore,
                    metadata=metadata
                )
            )
        
        for task in asyncio.as_completed(tasks):
            doc = await task 
    
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            yield doc # yield doc as soon as it is ready

async def get_files(loaders: Dict[str, BaseLoader], path: str | list=True, recursive=True) -> AsyncGenerator:
    """Get files from a directory or a list of files"""

    supported_types = loaders.keys()
    patterns = [f'**/*{type_}' for type_ in supported_types]

    if isinstance(path, list):
        for file_path in path:
            p = AsyncPath(file_path)
            if await p.is_file():
                type_ = p.suffix
                if type_ in supported_types: # check the file type
                    yield p
                else:
                    logger.warning(f"Unsupported file type: {type_}: {p.name} will not be indexed.")
    
    else:
        p = AsyncPath(path)
        if await p.is_dir():
            for pat in patterns:
                async for file in (p.rglob(pat) if recursive else p.glob(pat)):
                    yield file
        elif await p.is_file():
            type_ = p.suffix
            if type_ in supported_types: # check the file type
                yield p
            else:
                logger.warning(f"Unsupported file type: {type_}: {p.name} will not be indexed.")
        else:
            raise ValueError(f"Path {path} is neither a file nor a directory")
        
def get_loaders(config):
    
    loader_defaults = config["loader"]["file_loaders"]
    loader_classes = {}

    for type_, class_name in loader_defaults.items():
        cls = globals().get(class_name, None)
        if cls:
            loader_classes[f'.{type_}'] = cls
        else:
            raise ImportError(f"Class '{class_name}' not found. Program will crash if a file needs to be handled by this loader.")
        
    logger.debug(f"Loaders loaded: {loader_classes.keys()}")
    return loader_classes