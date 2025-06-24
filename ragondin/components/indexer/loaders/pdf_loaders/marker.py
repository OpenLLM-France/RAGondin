import asyncio
import gc
import re
import time
from pathlib import Path
from typing import Dict, Optional, Union

import ray
import torch
from config import load_config
from langchain_core.documents.base import Document
from marker.converters.pdf import PdfConverter
from utils.logger import get_logger

from ..base import BaseLoader

logger = get_logger()
config = load_config()


@ray.remote(num_gpus=config.loader.get("marker_num_gpus", 0))
class MarkerWorker:
    def __init__(self):
        import os

        from config import load_config
        from utils.logger import get_logger

        self.logger = get_logger()
        self.config = load_config()
        self.page_sep = "[PAGE_SEP]"

        self._workers = self.config.ray.get("max_tasks_per_worker", 2)
        self.maxtasksperchild = self.config.loader.get("marker_max_tasks_per_child", 10)

        self.converter_config = {
            "output_format": "markdown",
            "paginate_output": True,
            "page_separator": self.page_sep,
            "pdftext_workers": 1,
            "disable_multiprocessing": True,
        }
        if "RAY_ADDRESS" not in os.environ:
            os.environ["RAY_ADDRESS"] = "auto"
        self.pool = None
        self.init_resources()

    def init_resources(self):
        import torch.multiprocessing as mp
        from marker.models import create_model_dict

        if self.pool:
            self.logger.warning("Resetting multiprocessing pool")
            self.pool.close()
            self.pool.join()
            self.pool.terminate()
            self.pool = None
        self.model_dict = create_model_dict()
        for v in self.model_dict.values():
            if hasattr(v.model, "share_memory"):
                v.model.share_memory()

        try:
            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)
        except RuntimeError:
            self.logger.warning(
                "Process start method already set, using existing method"
            )

        self.logger.info(f"Initializing MarkerWorker with {self._workers} workers")
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(
            processes=self._workers,
            initializer=self._worker_init,
            initargs=(self.model_dict,),
            maxtasksperchild=self.maxtasksperchild,
        )

        self.logger.info("MarkerWorker initialized with multiprocessing pool")

    @staticmethod
    def _worker_init(model_dict):
        global worker_model_dict
        worker_model_dict = model_dict
        logger.debug("Worker initialized with model dictionary")

    @staticmethod
    def _process_pdf(file_path, config):
        global worker_model_dict

        try:
            logger.debug("Processing PDF", path=file_path)
            converter = PdfConverter(
                artifact_dict=worker_model_dict,
                config=config,
            )
            render = converter(file_path)
            return render
        except Exception:
            logger.exception("Error processing PDF", path=file_path)
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    async def process_pdf(self, file_path: str):
        config = self.converter_config.copy()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: self.pool.apply(self._process_pdf, (file_path, config))
        )
        return result.markdown, result.images

    def get_current_pool_size(self):
        return len([p for p in self.pool._pool if p.is_alive()])


@ray.remote
class MarkerPool:
    def __init__(self):
        from config import load_config
        from utils.logger import get_logger

        self.logger = get_logger()
        self.config = load_config()
        self.pool_size = config.loader.get("marker_pool_size", 2)
        self.actors = [MarkerWorker.remote() for _ in range(self.pool_size)]
        self._queue: asyncio.Queue[ray.actor.ActorHandle] = asyncio.Queue()

        for _ in range(self.pool_size):
            for actor in self.actors:
                self._queue.put_nowait(actor)

        self.logger.info(
            f"Marker pool: {self.pool_size} actors × {self.config.loader.get('marker_child_processes', 2)} slots = "
            f"{self.pool_size * self.config.loader.get('marker_child_processes', 2)} total concurrency"
        )

    async def process_pdf(self, file_path: str):
        self.logger.info("Processing PDF with MarkerWorker")

        # Wait until any slot is free
        actor = await self._queue.get()
        if actor:
            self.logger.info("MarkerWorker allocated")
        try:
            markdown, images = await actor.process_pdf.remote(file_path)
            return markdown, images
        except Exception as e:
            self.logger.exception(
                "Error processing PDF with MarkerWorker", error=str(e)
            )
            raise
        finally:
            await self._queue.put(actor)
            self.logger.debug("MarkerWorker returned to pool")


class MarkerLoader(BaseLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.page_sep = "[PAGE_SEP]"
        self.worker = ray.get_actor("MarkerPool", namespace="ragondin")

    async def aload_document(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict] = None,
        save_markdown: bool = False,
    ) -> Document:
        if metadata is None:
            metadata = {}

        file_path_str = str(file_path)
        start = time.time()

        try:
            markdown, images = await self.worker.process_pdf.remote(file_path_str)

            if not markdown:
                raise RuntimeError(f"Conversion failed for {file_path_str}")

            if self.config["loader"]["image_captioning"]:
                captions = await self._get_captions(images)
                for key, desc in captions.items():
                    markdown = markdown.replace(f"![]({key})", desc)
            else:
                logger.info("Image captioning disabled.")

            markdown = markdown.split(self.page_sep, 1)[1]
            markdown = re.sub(
                r"\{(\d+)\}" + re.escape(self.page_sep), r"[PAGE_\1]", markdown
            )
            markdown = markdown.replace("<br>", " <br> ").strip()

            doc = Document(page_content=markdown, metadata=metadata)

            if save_markdown:
                self.save_document(doc, file_path_str)

            duration = time.time() - start
            logger.info(f"Processed {file_path_str} in {duration:.2f}s")
            return doc

        except Exception:
            logger.exception("Error in aload_document", path=file_path_str)
            raise

    async def _get_captions(self, img_dict: dict) -> dict:
        if not img_dict:
            return {}

        tasks = [self.get_image_description(img) for img in img_dict.values()]
        keys = list(img_dict.keys())

        try:
            results = await asyncio.gather(*tasks)
            return dict(zip(keys, results))
        except Exception:
            logger.exception("Error during image captioning")
            raise
