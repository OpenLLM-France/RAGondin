import asyncio
import importlib
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional
import torch
from aiopath import AsyncPath
from langchain_core.documents.base import Document
from loguru import logger
import ray
from .base import BaseLoader

if not ray.is_initialized():
    ray.init(dashboard_host="0.0.0.0", ignore_reinit_error=True)


@ray.remote(concurrency_groups={"pdf": 3})
class DocumentProcessor:
    def __init__(self, loader_cls, kwargs):
        self.loader_cls = loader_cls
        self.kwargs = kwargs

    @ray.method(concurrency_group="pdf")
    async def process_document(self, path: str, metadata: Dict = {}):
        p = AsyncPath(path)
        logger.debug(f"LOADING: {p.name}")
        loader = self.loader_cls(**self.kwargs)
        # Process the document
        metadata.update({"page_sep": loader.page_sep})
        doc: Document = await loader.aload_document(
            file_path=path, metadata=metadata, save_md=True
        )
        logger.info(f"{p.name}: SERIALIZED")
        return doc


class DocSerializer:
    def __init__(self, data_dir=None, **kwargs) -> None:
        self.data_dir = data_dir
        self.kwargs = kwargs
        self.config = kwargs.get("config")
        self.loader_classes = get_loaders(self.config)

    async def serialize_document(self, path: str, metadata: Optional[Dict] = {}):
        p = AsyncPath(path)
        type_ = p.suffix
        loader_cls: BaseLoader = self.loader_classes.get(type_)
        sub_url_path = (
            Path(path).resolve().relative_to(self.data_dir)
        )  # for the static file server

        metadata = {
            "source": str(path),
            "file_name": p.name,
            "sub_url_path": str(sub_url_path),
            **metadata,
        }

        # Create a Ray actor for this document
        doc_processor = DocumentProcessor.remote(loader_cls, self.kwargs)

        # Special handling for PDF files that might use GPU
        if type_ == ".pdf":
            # Use Ray to handle GPU resource allocation
            future = doc_processor.process_document.options(
                concurrency_group="pdf"
            ).remote(path, metadata)
        else:
            future = doc_processor.process_document.remote(path, metadata)
        return future

    async def serialize_documents(
        self,
        path: str | Path | list[str],
        metadata: Optional[Dict] = {},
        recursive=True,
    ) -> AsyncGenerator[Document, None]:
        # Collect all files to process
        files = []
        async for file in get_files(self.loader_classes, path, recursive):
            files.append(file)

        # Start all tasks using Ray
        futures = []
        for file in files:
            future = await self.serialize_document(file, metadata=metadata)
            futures.append(future)

        # Yield documents as they complete
        while futures:
            # Use ray.wait to get completed tasks
            done_ids, futures = ray.wait(futures, num_returns=1)
            doc = await asyncio.to_thread(ray.get, done_ids[0])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            yield doc  # yield doc as soon as it is ready


async def get_files(
    loaders: Dict[str, BaseLoader], path: str | list = True, recursive=True
) -> AsyncGenerator:
    """Get files from a directory or a list of files"""

    supported_types = loaders.keys()
    patterns = [f"**/*{type_}" for type_ in supported_types]

    if isinstance(path, list):
        for file_path in path:
            p = AsyncPath(file_path)
            if await p.is_file():
                type_ = p.suffix
                if type_ in supported_types:  # check the file type
                    yield p
                else:
                    logger.warning(
                        f"Unsupported file type: {type_}: {p.name} will not be indexed."
                    )
    else:
        p = AsyncPath(path)
        if await p.is_dir():
            for pat in patterns:
                async for file in p.rglob(pat) if recursive else p.glob(pat):
                    yield file
        elif await p.is_file():
            type_ = p.suffix
            if type_ in supported_types:  # check the file type
                yield p
            else:
                logger.warning(
                    f"Unsupported file type: {type_}: {p.name} will not be indexed."
                )
        else:
            raise ValueError(f"Path {path} is neither a file nor a directory")


def get_loaders(config):
    loader_defaults = config["loader"]["file_loaders"]
    loader_classes = {}

    for type_, class_name in loader_defaults.items():
        try:
            module = importlib.import_module(f"components.indexer.loaders.{class_name}")
            cls = getattr(module, class_name)
            loader_classes[f".{type_}"] = cls
            logger.debug(f"Loaded {class_name} for {type_}")
        except (ModuleNotFoundError, AttributeError) as e:
            logger.error(f"Error loading {class_name}: {e}")

    logger.debug(f"Loaders loaded: {loader_classes.keys()}")
    return loader_classes
