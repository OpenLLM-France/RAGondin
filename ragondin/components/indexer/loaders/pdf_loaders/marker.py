import asyncio
from pathlib import Path
from typing import Dict, Optional, Union
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
import time
from langchain_core.documents.base import Document
from loguru import logger
from ..base import BaseLoader
from tqdm.asyncio import tqdm


class MarkerLoader(BaseLoader):
    """
    Loader for PDF files using the Marker library.
    """

    def __init__(self, page_sep: str = "[PAGE_SEP]", **kwargs):
        super().__init__(page_sep, **kwargs)
        self.page_sep = page_sep
        self._converter = None

    def _get_converter(self):
        """Lazily initialize the PDF converter."""
        if self._converter is None:
            self._converter = PdfConverter(
                artifact_dict=create_model_dict(),
                config={
                    "output_format": "markdown",
                    "paginate_output": True,
                    "page_separator": self.page_sep,
                },
            )
        return self._converter

    async def aload_document(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict] = None,
        save_md: bool = False,
    ) -> Document:
        if metadata is None:
            metadata = {}

        logger.info(f"Loading {file_path}")
        start = time.time()

        # Process PDF directly for this call
        converter = self._get_converter()

        # Run the conversion in a thread pool to avoid blocking the event loop
        try:
            render = await asyncio.to_thread(converter, str(file_path))
        except Exception as e:
            raise RuntimeError(f"Conversion failed: {str(e)}")

        text = render.markdown

        if self.config["loader"]["image_captioning"]:
            img_dict = render.images
            logger.info(f"Found {len(img_dict)} images in the document.")
            captions_dict = await self._get_captions(img_dict)
            for key, desc in captions_dict.items():
                tag = f"![]({key})"
                text = text.replace(tag, desc)
        else:
            logger.info("Image captioning disabled.")

        doc = Document(page_content=text, metadata=metadata)

        if save_md:
            self.save_document(doc, str(file_path))

        end = time.time()
        logger.info(f"Total time for file {file_path}: {end - start:.2f}s")
        return doc

    async def _get_captions(self, img_dict: dict):
        tasks = []
        keys = []

        for key, picture in img_dict.items():
            tasks.append(self.get_image_description(picture))
            keys.append(key)

        try:
            results = await tqdm.gather(*tasks, desc="Captioning images")
            assert len(img_dict.keys()) == len(results)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            raise

        result_dict = dict(zip(img_dict.keys(), results))
        return result_dict
