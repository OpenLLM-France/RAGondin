import asyncio
import time

from components.utils import SingletonMeta
from langchain_core.documents.base import Document
from loguru import logger
from tqdm.asyncio import tqdm

from .base import BaseLoader


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
                "output_format": "markdown",
                "paginate_output": True,
            },
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

    def __init__(
        self,
        page_sep: str = "------------------------------------------------\n\n",
        **kwargs,
    ) -> None:
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

    async def get_captions(self, img_dict):
        tasks = []

        for _, picture in img_dict.items():
            tasks.append(self.get_image_description(picture))
        try:
            results = await tqdm.gather(
                *tasks, desc="Captioning imgs"
            )  # asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()

            raise

        result_dict = dict(zip(img_dict.keys(), results))
        return result_dict
