import asyncio
import base64
from pathlib import Path
import re
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Dict, Optional, Union
from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from ...utils import vlmSemaphore, load_sys_template, load_config
from pathlib import Path


config = load_config()
prompts_dir = Path(config.paths.prompts_dir)
img_desc_prompt_path = prompts_dir / config.prompt["image_describer"]
IMAGE_DESCRIPTION_PROMPT = load_sys_template(img_desc_prompt_path)


class BaseLoader(ABC):
    def __init__(self, **kwargs) -> None:
        self.page_sep = "[PAGE_SEP]"
        self.config = kwargs.get("config")
        vlm_config = self.config.vlm
        model_settings = {
            "temperature": 0.2,
            "max_retries": 3,
            "timeout": 60,
        }
        settings: dict = vlm_config
        settings.update(model_settings)

        self.vlm_endpoint = ChatOpenAI(**settings).with_retry(stop_after_attempt=2)
        self.min_width_pixels = 0  # minimum width in pixels
        self.min_height_pixels = 0  # minimum height in pixels

    @abstractmethod
    async def aload_document(
        file_path: Union[str, Path],
        metadata: Optional[Dict] = None,
        save_markdown: bool = False,
    ):
        pass

    def save_document(self, doc: Document, path: str):
        path = re.sub(r"\..*", ".md", path)
        with open(path, "w", encoding="utf-8") as f:
            f.write(doc.page_content)
        logger.info(f"Document saved to {path}")

    async def get_image_description(
        self, image, semaphore: asyncio.Semaphore = vlmSemaphore
    ):
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
                            "url": f"data:image/png;base64,{img_b64}"  # f"{picture.image.uri.path}" #
                        },
                    },
                    {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
                ]
            )
            try:
                if width > self.min_width_pixels and height > self.min_height_pixels:
                    response = await self.vlm_endpoint.ainvoke([message])
                    image_description = response.content

            except Exception as e:
                logger.error(f"Error while generating image description: {e}")

            # Convert image path to markdown format and combine with description
            desc = f"""\n<image_description>\n{image_description}\n</image_description>\n"""
            return desc
