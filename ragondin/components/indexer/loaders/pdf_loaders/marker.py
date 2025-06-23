import asyncio
import gc
import re
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.multiprocessing as mp
from langchain_core.documents.base import Document
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from tqdm.asyncio import tqdm
from utils.logger import get_logger

from ..base import BaseLoader

logger = get_logger()


class MarkerLoader(BaseLoader):
    """
    Loader for PDF files using the Marker library.
    Implements shared resource management for parallel processing.
    """

    # Class variables for resource sharing
    _model_dict = None
    _pool = None
    _initialized = False
    _pool_lock = threading.RLock()  # Thread-safe lock for pool operations

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._workers = self.config.ray.get("max_tasks_per_worker", 2)
        self.maxtasksperchild = self.config.loader.get("marker_max_tasks_per_child", 10)
        logger.info("Initializing MarkerLoader", workers=self._workers)
        self._converter_config = {
            "output_format": "markdown",
            "paginate_output": True,
            "page_separator": self.page_sep,
            "pdftext_workers": 1,  # force single-threaded processing in the underlying pdftext lib. This is because RAY actors are daemon processes and it doesn't allow child processes.
            "disable_multiprocessing": True,  # We manage our own multiprocessing
        }

        # Initialize shared resources on first instance
        self._ensure_resources_initialized()

    def _ensure_resources_initialized(self):
        """Initialize shared resources if not already done"""
        if not MarkerLoader._initialized:
            with MarkerLoader._pool_lock:
                if not MarkerLoader._initialized:  # Double-check under lock
                    self._initialize_shared_resources()
                    MarkerLoader._initialized = True

    def _initialize_shared_resources(self):
        import os

        if "RAY_ADDRESS" not in os.environ:
            os.environ["RAY_ADDRESS"] = "auto"
        """Initialize model dictionary and worker pool once for all instances"""
        # Initialize the model dictionary
        MarkerLoader._model_dict = create_model_dict()

        # Share memory for models supporting it
        for k, v in MarkerLoader._model_dict.items():
            if hasattr(v.model, "share_memory"):
                v.model.share_memory()

        # Initialize the worker pool
        logger.info("Creating marker worker pool", workers=self._workers)
        ctx = mp.get_context("spawn")
        MarkerLoader._pool = ctx.Pool(
            processes=self._workers,
            initializer=self._worker_init,  # Note: Using class method directly
            initargs=(MarkerLoader._model_dict,),
            maxtasksperchild=self.maxtasksperchild,  # Restart workers periodically to prevent memory leaks
        )

    @staticmethod
    def _worker_init(model_dict):
        """Initialize each worker with model references"""
        global worker_model_dict
        worker_model_dict = model_dict
        logger.debug("Worker initialized with model dictionary")

    @staticmethod
    def _process_pdf(file_path, config):
        """Worker function to process a single PDF"""
        global worker_model_dict

        try:
            logger.debug("Processing PDF", path=file_path)
            converter = PdfConverter(
                artifact_dict=worker_model_dict,
                config=config,
            )
            render = converter(file_path)
            logger.debug("PDF processing completed", path=file_path)

            # Explicit cleanup
            del converter
            return render
        except Exception:
            logger.exception("Error processing PDF", path=file_path)
            raise
        finally:
            # Force garbage collection
            gc.collect()
            # Ensure CUDA memory is cleaned up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    async def aload_document(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict] = None,
        save_markdown: bool = False,
    ) -> Document:
        """
        Load a single document asynchronously.
        Multiple concurrent calls will be processed in parallel using the shared worker pool.
        """
        if metadata is None:
            metadata = {}

        file_path_str = str(file_path)
        start_time = time.time()

        # Configure for this specific document
        config = self._converter_config.copy()

        try:
            # Run the processing in the pool
            # Note: apply_async would be more efficient but needs different async handling
            result = await asyncio.get_event_loop().run_in_executor(
                None,  # Use default executor
                lambda: MarkerLoader._pool.apply(
                    MarkerLoader._process_pdf,
                    (file_path_str, config),  # Pass arguments directly, not as tuple
                ),
            )

            if result is None:
                logger.error("PDF conversion returned None", path=file_path_str)
                raise RuntimeError(f"Conversion failed for {file_path_str}")

            text: str = result.markdown

            # Process images if needed
            if self.config["loader"]["image_captioning"]:
                img_dict = result.images
                logger.info("Captioning images", image_count=len(img_dict))
                captions_dict = await self._get_captions(img_dict)
                for key, desc in captions_dict.items():
                    tag = f"![]({key})"
                    text = text.replace(tag, desc)
            else:
                logger.info("Image captioning disabled.")

            # skips any empty string before first page
            text = text.split(self.page_sep, 1)[1]
            text = re.sub(r"\{(\d+)\}" + re.escape(self.page_sep), r"[PAGE_\1]", text)
            text = text.replace("<br>", " <br> ")
            text = text.strip()
            # text = text.replace("<br>", " ")

            doc = Document(page_content=text, metadata=metadata)

            if save_markdown:
                self.save_document(doc, file_path_str)

            end_time = time.time()
            logger.info(
                f"Total time for file {file_path_str}: {end_time - start_time:.2f}s"
            )

            # Clean up CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            return doc

        except Exception:
            logger.exception("Error in aload_document", path=file_path_str)
            raise

    async def _get_captions(self, img_dict: dict):
        """Get captions for images"""
        if not img_dict:
            logger.debug("No images to caption")
            return {}

        tasks = []
        keys = []
        for key, picture in img_dict.items():
            tasks.append(self.get_image_description(picture))
            keys.append(key)

        try:
            results = await tqdm.gather(*tasks, desc="Captioning images")
            assert len(keys) == len(results), "Mismatch between keys and results count"
        except asyncio.CancelledError:
            logger.warning("Image captioning tasks cancelled")
            for task in tasks:
                task.cancel()
            raise
        except Exception:
            logger.exception("Error in _get_captions")
            raise

        result_dict = dict(zip(keys, results))
        return result_dict

    @classmethod
    def cleanup_resources(cls):
        """Manually clean up shared resources - can be called when needed"""
        with cls._pool_lock:
            if cls._pool:
                logger.debug("Cleaning up worker pool")
                cls._pool.close()
                cls._pool.join()
                cls._pool = None
                cls._model_dict.clear()
                cls._model_dict = None
                cls._initialized = False
                logger.debug("Worker pool cleanup complete")

                # Force garbage collection and CUDA cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    # Synchronize to ensure cleanup is complete
                    torch.cuda.synchronize()

    def __del__(self):
        """Attempt cleanup on garbage collection"""
        # No cleanup here to avoid issues with multiple instances
        # Resources should be cleaned up explicitly when the application shuts down
        pass
