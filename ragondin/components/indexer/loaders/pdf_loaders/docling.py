import asyncio
import torch
from components.utils import SingletonMeta
from docling.datamodel.document import ConversionResult
from docling_core.types.doc.document import PictureItem
from langchain_core.documents.base import Document
from loguru import logger
from tqdm.asyncio import tqdm
from ..base import BaseLoader
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
    def __init__(self):
        img_scale = 1
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
            device=AcceleratorDevice.AUTO
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
    def __init__(self, page_sep="[PAGE_SEP]", **kwargs):
        super().__init__(page_sep, **kwargs)
        self.converter = DoclingConverter()

    async def aload_document(self, file_path, metadata, save_markdown=False):
        with torch.no_grad():
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
        if save_markdown:
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
