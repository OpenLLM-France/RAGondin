import asyncio
import time

import torch
from components.utils import SingletonMeta
from langchain_core.documents.base import Document
from loguru import logger
from tqdm.asyncio import tqdm
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from .base import BaseLoader
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor


if torch.cuda.is_available():
    mp.set_start_method("spawn", force=True)


# TODO: Pagination correct number and Image resolution to avoid captioning small images
class MarkerConverter(metaclass=SingletonMeta):
    def __init__(self, page_sep) -> None:
        self.model_dict = create_model_dict()
        for k, v in self.model_dict.items():
            v.model.share_memory()

        self.converter = PdfConverter(
            artifact_dict=self.model_dict,
            config={
                "output_format": "markdown",
                "paginate_output": True,
                "page_separator": page_sep,
            },
        )

        # Create a thread pool executor to reuse threads across conversions.
        self.executor = ThreadPoolExecutor(
            max_workers=12
        )  # Adjust max_workers as needed.

    async def convert_to_md(self, file_path):
        loop = asyncio.get_running_loop()
        # Use process pool to run the conversion in a separate process
        output = await loop.run_in_executor(
            self.executor,  # Using default executor
            self.converter,
            str(file_path),
        )
        return output


class MarkerLoader(BaseLoader):
    """
    MarkerLoader is a class responsible for loading and converting documents into markdown format,
    with optional image captioning using a language model.
    Attributes:
        page_sep (str): Separator used for pages in the document.
        converter (MarkerConverter): Instance of MarkerConverter for converting documents to markdown.
    """

    def __init__(
        self,
        page_sep: str = "[PAGE_SEP]",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.page_sep = page_sep
        self.converter = MarkerConverter(page_sep=page_sep)

    @classmethod
    def destroy(cls):
        if MarkerConverter in SingletonMeta._instances:
            del SingletonMeta._instances[MarkerConverter]

    async def aload_document(self, file_path, metadata=None, save_md=False):
        file_path = str(file_path)
        with torch.no_grad():
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
                tag = f"![]({key})"
                text = text.replace(tag, desc)
        else:
            logger.info("Image captioning disabled. Ignoring images.")

        end = time.time()
        logger.info(f"Total conversion time for file {file_path}: {end - start:.2f} s.")

        doc = Document(page_content=text, metadata=metadata)

        if save_md:
            self.save_document(Document(page_content=text), str(file_path))

        return doc

    async def get_captions(self, img_dict: dict):
        tasks = []
        keys = []

        for key, picture in img_dict.items():
            tasks.append(self.get_image_description(picture))
            keys.append(key)
        try:
            results = await tqdm.gather(*tasks, desc="Captioning imgs")
            assert len(img_dict.keys()) == len(results)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()

            raise

        result_dict = dict(zip(img_dict.keys(), results))
        return result_dict
