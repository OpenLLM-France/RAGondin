import asyncio
from components.utils import SingletonMeta, SingletonABCMeta
from docling.datamodel.document import ConversionResult
from docling_core.types.doc.document import PictureItem
from langchain_core.documents.base import Document
from loguru import logger
from tqdm.asyncio import tqdm
from .base import BaseLoader
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


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
        img_scale = 2
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            generate_picture_images=True,
            images_scale=img_scale,
            # generate_table_images=True,
            # generate_page_images=True
        )
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True, mode=TableFormerMode.ACCURATE
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
        o = await asyncio.to_thread(self.converter.convert, str(file_path))
        return o


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

    def __init__(self, page_sep: str = "[PAGE_SEP]", **kwargs) -> None:
        super().__init__(**kwargs)
        self.page_sep = page_sep
        self.converter = DoclingConverter()

    async def aload_document(self, file_path, metadata, save_md=False):
        result = await self.converter.convert_to_md(file_path)

        n_pages = len(result.pages)
        s = f"{self.page_sep}".join(
            [
                result.document.export_to_markdown(page_no=i)
                for i in range(1, n_pages + 1)
            ]
        )
        enriched_content = s
        if self.config.loader["image_captioning"]:
            pictures = result.document.pictures
            descriptions = await self.get_captions(pictures)
            for description in descriptions:
                enriched_content = enriched_content.replace(
                    "<!-- image -->", description, 1
                )
        else:
            logger.info("Image captioning disabled. Ignoring images.")

        doc = Document(page_content=enriched_content, metadata=metadata)
        if save_md:
            self.save_document(Document(page_content=enriched_content), str(file_path))
        return doc

    async def get_captions(self, pictures: list[PictureItem]):
        tasks = []
        for picture in pictures:
            tasks.append(self.get_image_description(picture.image.pil_image))
        try:
            results = await tqdm.gather(*tasks, desc="Captioning imgs")
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            raise
        return results

    async def convert_to_md(self, file_path) -> ConversionResult:
        return await asyncio.to_thread(self.converter.convert, str(file_path))

    async def parse(self, file_path, page_seperator="[PAGE_SEP]"):
        result = await self.convert_to_md(file_path)
        n_pages = len(result.pages)
        s = f"{page_seperator}".join(
            [
                result.document.export_to_markdown(page_no=i)
                for i in range(1, n_pages + 1)
            ]
        )
        pictures = result.document.pictures
        descriptions = await self.get_captions(pictures)
        enriched_content = s
        for description in descriptions:
            enriched_content = enriched_content.replace(
                "<!-- image -->", description, 1
            )
        return enriched_content
