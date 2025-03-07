from abc import abstractmethod, ABC
import asyncio
import os
from collections import defaultdict
import gc
from langchain_openai import ChatOpenAI
from components.utils import SingletonMeta
from pydub import AudioSegment
import torch
from pathlib import Path
from typing import AsyncGenerator, Optional, Dict
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

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

import pymupdf4llm
from loguru import logger
from aiopath import AsyncPath
from typing import Dict
from docling_core.types.doc.document import PictureItem
from docling.datamodel.document import ConversionResult

from tqdm.asyncio import tqdm

import re
import base64
from io import BytesIO

import time

# from langchain_community.document_loaders import UnstructuredXMLLoader, PyPDFLoader
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.document_loaders.text import TextLoader
# from langchain_community.document_loaders import UnstructuredHTMLLoader

class BaseLoader(ABC):
    @abstractmethod
    async def aload_document(self, file_path, sub_url_path: str =''):
        pass

    def save_document(self, doc: Document, path: str):
        path = re.sub(r'\..*', '.md', path)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(doc.page_content)



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
    def __init__(self, llm_config=None):
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
        
        pipeline_options = PdfPipelineOptions(
            do_ocr = True,
            do_table_structure = True,
            generate_picture_images=True,
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

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
                )
            }
        )

    async def describe_imgage(self, idx, picture: PictureItem, semaphore: asyncio.Semaphore):
        async with semaphore:
            page_no = picture.prov[0].page_no    
            img = picture.image.pil_image
            img_b64 = picture._image_to_base64(pil_image=img)
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
                        "text": """Provide a concise complete, structured and precise description of this image or figure in the same language (french) as its content. If the image contains tables, render them in markdown."""
                    }
                ]
            )
            try:
                if (img.width > self.min_width_pixels and img.height > self.min_height_pixels):
                    response = await self.vlm_endpoint.ainvoke([message])
                    image_description = response.content
                #     img.save(f"./temp_img/figure_page_{page_no}_{img.width}X{img.height}.png")
                # else:
                #     img.save(f"./temp_img/no_figure_page_{page_no}_{img.width}X{img.height}.png")

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

    async def get_captions(self, pictures: list[PictureItem], n_semaphores=10):
        semaphore = asyncio.Semaphore(n_semaphores)
        tasks = []

        for idx, picture in enumerate(pictures):
            tasks.append(
                self.describe_imgage(idx, picture, semaphore)
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
        # TODO: get rid of blocking tasks
        result = await self.convert_to_md(file_path)
        n_pages = len(result.pages)
        s = f'{page_seperator}'.join([result.document.export_to_markdown(page_no=i) for i in range(1, n_pages+1)])
        pictures = result.document.pictures
        descriptions = await self.get_captions(pictures, n_semaphores=6)
        enriched_content = s
        for description in descriptions:
            enriched_content = enriched_content.replace('<!-- image -->', description, 1)

        return enriched_content


class DoclingLoader(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        self.page_sep = page_sep
        llm_config = kwargs.get('llm_config')
        self.converter = DoclingConverter(llm_config=llm_config)
    
    async def aload_document(self, file_path, metadata: dict = None):
        content = await self.converter.parse(file_path, page_seperator=self.page_sep)
        return Document(
            page_content=content, 
            metadata=metadata
        )
    
class MarkerConverter(metaclass=SingletonMeta):
    def __init__(self) -> None:
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
    def __init__(self, page_sep: str='------------------------------------------------\n\n', **kwargs) -> None:
        self.page_sep = page_sep
        llm_config = kwargs.get('llm_config')
        self.converter = MarkerConverter()
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

    async def get_image_description(self, image, semaphore: asyncio.Semaphore):
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
                #     img.save(f"./temp_img/figure_page_{page_no}_{img.width}X{img.height}.png")
                # else:
                #     img.save(f"./temp_img/no_figure_page_{page_no}_{img.width}X{img.height}.png")

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

    async def aload_document(self, file_path, sub_url_path: str = ''):
        file_path = str(file_path)
        logger.info(f"Loading {file_path}")
        start = time.time()
        render = await self.converter.convert_to_md(file_path)
        conversion_time = time.time() - start
        logger.info(f"Markdown conversion time: {conversion_time:.2f} s.")

        text = render.markdown

        # Parse and replace images with llm based descriptions
        # Find all instances of markdown image tags
        img_dict = render.images
        logger.info(f"Found {len(img_dict)} images in the document.")

        captions_dict = await self.get_captions(img_dict)

        for key, desc in captions_dict.items():
            tag = f'![]({key})'
            text = text.replace(tag, desc)

        end = time.time()
        logger.info(f"Total conversion time for file {file_path}: {end - start:.2f} s.")

        return Document(
            page_content=text, 
            metadata={
                'source': str(file_path),
                'sub_url_path': sub_url_path,
                'page_sep': self.page_sep,
            }
        )
    
    async def get_captions(self, img_dict, n_semaphores=10):
        semaphore = asyncio.Semaphore(n_semaphores)
        tasks = []

        for _, picture in img_dict.items():
            tasks.append(
                self.get_image_description(picture, semaphore)
            )
        try:
            results = await tqdm.gather(*tasks, desc='Captioning imgs')  # asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            
            raise

        result_dict = dict(zip(img_dict.keys(), results))

        return result_dict

class MarkerLoader(BaseLoader):
    def __init__(self, page_sep: str='------------------------------------------------\n\n', **kwargs) -> None:
        self.page_sep = page_sep
        self.converter = PdfConverter(
            artifact_dict=create_model_dict(),
            config={
                    'output_format': 'markdown',
                    'paginate_output': True,
                }
        )
    
    async def aload_document(self, file_path, sub_url_path: str = ''):
        file_path = str(file_path)
        logger.info(f"Loading {file_path}")
        render = self.converter(file_path)

        # Get enclosing folder
        folder = file_path.replace('.pdf', '')
        os.makedirs(folder, exist_ok=True)

        # Save document
        with open(file_path.replace('.pdf', '/markdown.md'), 'w', encoding='utf-8') as f:
            f.write(render.markdown)
        
        # Save images
        img_dict = render.images
        for key, image in img_dict.items():
            image.save(os.path.join(folder, key))



        return Document(
            page_content=render.markdown, 
            metadata={
                'source': str(file_path),
                'sub_url_path': sub_url_path,
                'page_sep': self.page_sep
            }
        )


class DocSerializer:
    def __init__(self, data_dir=None, **kwargs) -> None:
        self.data_dir = data_dir
        self.kwargs = kwargs
    
    # TODO: Add delete class obj
    async def serialize_document(self, path: str, semaphore: asyncio.Semaphore, metadata: Optional[Dict] = {}):
        async with semaphore:
            p = AsyncPath(path)
            type_ = p.suffix
            loader_cls: BaseLoader = LOADERS.get(type_)
            logger.debug(f'LOADING: {p.name}')
            sub_url_path = Path(path).absolute().relative_to(self.data_dir)
            loader = loader_cls(**self.kwargs)  # Propagate kwargs here!
            metadata={**{'source': str(path),'file_name': p.name ,'sub_url_path': str(sub_url_path),'page_sep': loader.page_sep},**metadata}
            doc: Document = await loader.aload_document(
                file_path=path,
                metadata=metadata
            )
            logger.debug(f"{p.name}: SERIALIZED")
            return doc

    async def serialize_documents(self, path: str | Path | list[str], metadata: Optional[Dict] = {}, recursive=True, n_concurrent_ops=3) -> AsyncGenerator[Document, None]:
        semaphore = asyncio.Semaphore(n_concurrent_ops)
        tasks = []
        async for file in get_files(path, recursive):
            tasks.append(
                self.serialize_document(
                    file,
                    semaphore=semaphore,
                    metadata=metadata
                )
            )
        
        for task in asyncio.as_completed(tasks):
            doc = await task 
            yield doc # yield doc as soon as it is ready

async def get_files(path: str | list=True, recursive=True) -> AsyncGenerator:
    """Get files from a directory or a list of files"""

    if isinstance(path, list):
        for file_path in path:
            p = AsyncPath(file_path)
            if await p.is_file():
                type_ = p.suffix
                if type_ in SUPPORTED_TYPES: # check the file type
                    yield p
                else:
                    logger.warning(f"Unsupported file type: {type_}: {p.name} will not be indexed.")
    
    else:
        p = AsyncPath(path)
        if await p.is_dir():
            for pat in PATTERNS:
                async for file in (p.rglob(pat) if recursive else p.glob(pat)):
                    yield file
        elif await p.is_file():
            type_ = p.suffix
            if type_ in SUPPORTED_TYPES: # check the file type
                yield p
            else:
                logger.warning(f"Unsupported file type: {type_}: {p.name} will not be indexed.")
        else:
            raise ValueError(f"Path {path} is neither a file nor a directory")
            


# TODO create a Meta class that aggregates registery of supported documents from each child class
LOADERS: Dict[str, BaseLoader] = {
    '.pdf': MarkerLoader, # CustomPyMuPDFLoader, # 
    '.docx': CustomDocLoader,
    '.doc': CustomDocLoader,
    '.odt': CustomDocLoader,

    # '.mp4': VideoAudioLoader,
    # '.pptx': CustomPPTLoader,
    # '.txt': CustomTextLoader,
    #'.html': CustomHTMLLoader
}

SUPPORTED_TYPES = LOADERS.keys()
PATTERNS = [f'**/*{type_}' for type_ in SUPPORTED_TYPES]

# if __name__ == "__main__":
#     async def main():
#         loader = DocSerializer()
#         dir_path = "../../data/"  # Replace with your actual directory path
#         docs = loader.serialize_documents(dir_path)
#         async for d in docs:
#             pass

                    
#     asyncio.run(main())


# uv run ./manage_collection.py -l  -o chunker.contextual_retrieval=false