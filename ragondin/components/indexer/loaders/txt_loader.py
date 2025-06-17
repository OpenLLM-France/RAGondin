"""
Text file loader implementation.
"""

import asyncio
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Union

import aiohttp
from components.indexer.loaders.base import BaseLoader
from langchain_community.document_loaders import TextLoader as LangchainTextLoader
from langchain_core.documents.base import Document
from PIL import Image
from utils.logger import get_logger

logger = get_logger()


class TextLoader(BaseLoader):
    """
    Loader for plain text files (.txt).
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    async def aload_document(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict] = None,
        save_markdown: bool = False,
    ) -> Document:
        if metadata is None:
            metadata = {}

        path = Path(file_path)
        loader = LangchainTextLoader(file_path=str(path), autodetect_encoding=True)

        # Load document segments asynchronously
        doc_segments = await loader.aload()

        # Create final document
        s = ""
        for page_num, segment in enumerate(doc_segments, start=1):
            s = segment.page_content.strip()

        doc = Document(
            page_content=s,
            metadata=metadata,
        )

        # Save if requested
        if save_markdown:
            self.save_document(doc, str(path))

        return doc


class MarkdownLoader(BaseLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._inline_img_pattern = re.compile(r"!\[(.*?)\]\((https?://.*?)\)")

    async def _fetch_image(self, url: str) -> Optional[Image.Image]:
        """Fetch image from URL using aiohttp."""
        try:
            logger.debug(f"Fetching image from URL: {url}")
            timeout = aiohttp.ClientTimeout(total=30)  # Add timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        return Image.open(BytesIO(content))
                    else:
                        logger.warning(
                            f"Failed to fetch image: HTTP {response.status} for {url}"
                        )
                        return None
        except asyncio.TimeoutError:
            logger.warning("Timeout fetching image", url=url)
            return None
        except Exception:
            logger.exception("Failed to fetch or open image", url=url)
            return None

    async def _process_image_with_description(self, alt: str, url: str):
        """Process a single image and return its markdown syntax and description."""
        markdown_syntax = f"![{alt}]({url})"
        img = await self._fetch_image(url)

        if img:
            try:
                description = await self.get_image_description(img)
                return markdown_syntax, description or alt  # Fallback to alt text
            except Exception:
                logger.exception("Failed to get image description", url=url)
                return markdown_syntax, alt  # Fallback to alt text
        else:
            return markdown_syntax, alt  # Fallback to alt text

    async def _get_url_images_with_descriptions(self, text: str) -> dict:
        """Process all images in parallel and get their descriptions."""
        matches = self._inline_img_pattern.findall(text)
        logger.debug("Found inline images", image_count=len(matches))

        if not matches:
            return {}

        # Process all images concurrently
        tasks = [self._process_image_with_description(alt, url) for alt, url in matches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and convert results to dictionary
        descriptions_dict = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Image processing failed: {result}")
                continue
            descriptions_dict[result[0]] = result[1]

        logger.debug(
            "Successfully processed images", Image_count=len(descriptions_dict)
        )
        return descriptions_dict

    async def aload_document(self, file_path, metadata=None, save_markdown=False):
        """
        Load and process a markdown document.

        Args:
            file_path: Path to the markdown file
            metadata: Optional metadata for the document
            save_markdown: Whether to save the processed markdown

        Returns:
            Processed Document object
        """
        if metadata is None:
            metadata = {}

        path = Path(file_path)
        logger.debug("Loading markdown file", path=path)

        try:
            raw_text = path.read_text(encoding="utf-8")
            logger.debug("Read markdown file", path=path, length=len(raw_text))
        except Exception:
            logger.exception("Failed to read markdown file", path=path)
            raise

        # Process all images concurrently
        image_descriptions = await self._get_url_images_with_descriptions(raw_text)

        # Replace image references with descriptions
        clean_text = raw_text
        if image_descriptions:
            logger.debug(
                "Replacing image references", image_count=len(image_descriptions)
            )
            for md_syntax, description in image_descriptions.items():
                clean_text = clean_text.replace(md_syntax, description)
        else:
            logger.debug("No images found to process")

        # Create document with processed content
        doc = Document(
            page_content=clean_text,
            metadata=metadata,
        )

        if save_markdown:
            logger.debug("Saving processed markdown", path=path)
            self.save_document(doc, str(path))

        return doc
