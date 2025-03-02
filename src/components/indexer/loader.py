from abc import abstractmethod, ABC
import asyncio
import os
from collections import defaultdict
import gc
from langchain_openai import ChatOpenAI
from src.components.utils import SingletonMeta
from pydub import AudioSegment
import torch
from pathlib import Path
from typing import AsyncGenerator
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

# from langchain_community.document_loaders import UnstructuredXMLLoader, PyPDFLoader
# from langchain_community.document_loaders.csv_loader import CSVLoader
# from langchain_community.document_loaders.text import TextLoader
# from langchain_community.document_loaders import UnstructuredHTMLLoader

class BaseLoader(ABC):
    @abstractmethod
    async def aload_document(self, file_path, sub_url_path: str =''):
        pass


class Custompymupdf4llm(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', config=None, **kwargs) -> None:
        self.page_sep = page_sep
    
    async def aload_document(self, file_path, sub_url_path: str =''):
        pages = pymupdf4llm.to_markdown(
            file_path,
            write_images=False,
            page_chunks=True,
        )
        page_content = f'{self.page_sep}'.join([p['text'] for p in pages])
        return Document(
            page_content=page_content, 
            metadata={
                'source': str(file_path),
                'sub_url_path': sub_url_path,
                'page_sep': self.page_sep
            }
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

    async def aload_document(self, file_path, sub_url_path: str = ''):
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
            metadata={
                'source': str(file_path),
                'sub_url_path': sub_url_path,
                'page_sep': self.page_sep
            }
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
        self.loader_args = loader_args
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
    
    async def aload_document(self, file_path, sub_url_path: str = ''):
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
            metadata={
                'source': str(file_path),
                'sub_url_path': sub_url_path,
                'page_sep': self.page_sep
            }
        )

class CustomPyMuPDFLoader(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        self.page_sep = page_sep

    async def aload_document(self, file_path, sub_url_path: str = ''):
        loader = PyMuPDFLoader(
            file_path=Path(file_path),
        )
        pages = await loader.aload()
        return Document(
            page_content=f'{self.page_sep}'.join([p.page_content for p in pages]), 
            metadata={
                'source': str(file_path),
                'sub_url_path': sub_url_path,
                'page_sep': self.page_sep
            }
        )

class CustomTextLoader(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        self.page_sep = page_sep

    async def aload_document(self, file_path, sub_url_path: str = ''):
        path = Path(file_path)
        loader = TextLoader(file_path=str(path), autodetect_encoding=True)
        doc = await loader.aload()
        return Document(
            page_content=f'{self.page_sep}'.join([p.page_content for p in doc]), 
            metadata={
                'source': str(file_path),
                'sub_url_path': sub_url_path,
                'page_sep': self.page_sep
            }
        )
    

class CustomHTMLLoader(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        self.page_sep = page_sep

    async def aload_document(self, file_path, sub_url_path: str = ''):
        path = Path(file_path)
        loader = UnstructuredHTMLLoader(file_path=str(path), autodetect_encoding=True)
        doc = await loader.aload()
        return Document(
            page_content=f'{self.page_sep}'.join([p.page_content for p in doc]), 
            metadata={
                'source': str(file_path),
                'sub_url_path': sub_url_path,
                'page_sep': self.page_sep
            }
        )


class CustomDocLoader(BaseLoader):
    doc_loaders = {
            ".docx": UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.odt': UnstructuredODTLoader
        }
    
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        self.page_sep = page_sep
    
    
    async def aload_document(self, file_path, sub_url_path: str = ''):
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
            metadata={
                'source': str(file_path),
                'sub_url_path': sub_url_path,
                'page_sep': self.page_sep
            }
        )
    

class DoclingConverter: # (metaclass=SingletonMeta):
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
            generate_picture_images=True
        )
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True,
            mode=TableFormerMode.ACCURATE
        )

        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=6, device=AcceleratorDevice.AUTO
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
                        "text": "Provide a complete, structured and precise description of this image or figure in the same language (french) as its content."
                    }
                ]
            )
            try:
                if (img.width > self.min_width_pixels and img.height > self.min_height_pixels):
                    response = await self.vlm_endpoint.ainvoke([message])
                    image_description = response.content
                    # img.save(f"./temp_img/figure_page_{page_no}_{img.width}X{img.height}.png")
                # else:
                #     img.save(f"./temp_img/figure_page_{page_no}_{img.width}X{img.height}.png")

            except Exception as e:
                print(f"Failed to describe this image: {e}")

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
            results = await tqdm.gather(*tasks)  # asyncio.gather(*tasks)
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
        descriptions = await self.get_captions(pictures)

        enriched_content = s
        for description in descriptions:
            enriched_content = enriched_content.replace('<!-- image -->', description, 1)
        
        saving_path = Path(file_path).with_suffix('.md')
        t = enriched_content.replace(page_seperator, '\n')
        with open(saving_path, 'w', encoding='utf-8') as f:
            f.write(t)

        return enriched_content


class DoclingLoader(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', **kwargs) -> None:
        self.page_sep = page_sep
        llm_config = kwargs.get('llm_config')
        self.converter = DoclingConverter(llm_config=llm_config)
    
    async def aload_document(self, file_path, sub_url_path: str = ''):
        content = await self.converter.parse(file_path, page_seperator=self.page_sep)
        return Document(
            page_content=content, 
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
    async def serialize_document(self, path: str):
        p = AsyncPath(path)
        if await p.is_file():
            type_ = p.suffix
            loader_cls: BaseLoader = LOADERS.get(type_)
            logger.info(f'Loading {type_} files.')

            loader = loader_cls(**self.kwargs)  # Propagate kwargs here!
            doc: Document = await loader.aload_document(
                file_path=path,
                sub_url_path=Path(path).absolute().relative_to(self.data_dir) # for the static file server
            )
            yield doc

    async def serialize_documents(self, path: str | Path | list[str], recursive=True) -> AsyncGenerator[Document, None]:
        if isinstance(path, list): # list of file paths
            for file_path in path:
                async for doc in self.serialize_document(file_path):
                    yield doc
        else:
            p = AsyncPath(path)
            if await p.is_file():
                async for doc in self.serialize_document(path):
                    yield doc

            is_dir = await p.is_dir()
            if is_dir:
                for type, loader_cls in LOADERS.items():
                    pattern = f"**/*{type}"
                    logger.info(f'Loading {type} files.')
                    files = get_files(path, pattern, recursive) 
                    
                    async for file in files:
                        loader = loader_cls(**self.kwargs)
                        doc: Document = await loader.aload_document(
                            file_path=file,
                            sub_url_path=Path(file).absolute().relative_to(self.data_dir) # for the static file server
                        )
                        print(f"==> Serialized: {str(file)}")
                        yield doc
                    


async def get_files(path, pattern, recursive) -> AsyncGenerator:
    p = AsyncPath(path)
    async for file in (p.rglob(pattern) if recursive else p.glob(pattern)):
        yield file



# TODO create a Meta class that aggregates registery of supported documents from each child class
LOADERS: Dict[str, BaseLoader] = {
    '.pdf': DoclingLoader, # CustomPyMuPDFLoader, # 
    '.docx': CustomDocLoader,
    '.doc': CustomDocLoader,
    '.odt': CustomDocLoader,

    # '.mp4': VideoAudioLoader,
    '.pptx': CustomPPTLoader,
    '.txt': CustomTextLoader,
    #'.html': CustomHTMLLoader
}


# if __name__ == "__main__":
#     async def main():
#         loader = DocSerializer()
#         dir_path = "../../data/"  # Replace with your actual directory path
#         docs = loader.serialize_documents(dir_path)
#         async for d in docs:
#             pass

                    
#     asyncio.run(main())