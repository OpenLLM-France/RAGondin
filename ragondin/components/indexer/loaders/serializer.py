"""
Document serialization manager.

This module contains the main DocSerializer class that handles
the loading and serialization of documents from various file types.
"""

import gc
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from aiopath import AsyncPath
from langchain_core.documents.base import Document
from loguru import logger

from . import get_loader_classes


class DocSerializer:
    def __init__(self, data_dir=None, **kwargs) -> None:
        self.data_dir = data_dir
        self.kwargs = kwargs
        self.config = kwargs.get("config", {})

        # Initialize loader classes:
        self.loader_classes = get_loader_classes(config=self.config)

    async def serialize_document(
        self,
        path: Union[str, Path],
        metadata: Optional[Dict] = {},
    ) -> Document:
        if metadata is None:
            metadata = {}

        p = AsyncPath(path)
        file_ext = p.suffix

        # Get appropriate loader for the file type
        loader_cls = self.loader_classes.get(file_ext)
        if loader_cls is None:
            logger.info(f"No loader available for {p.name}")
            return None

        logger.debug(f"Loading document: {p.name}")
        loader = loader_cls(**self.kwargs)  # Propagate kwargs here!

        metadata = {"page_sep": loader.page_sep, **metadata}

        try:
            # Load the doc
            doc: Document = await loader.aload_document(
                file_path=path, metadata=metadata, save_md=False
            )

            # Clean up resources for specific loader types
            if hasattr(loader, "shutdown") and callable(loader.shutdown):
                await loader.shutdown()

            # Clean up resources
            del loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            return doc
        except Exception as e:
            logger.error(f"Error loading document {path}: {e}")
            raise e
