import asyncio
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
import uuid
import time
from langchain_core.documents.base import Document
from loguru import logger
from .base import BaseLoader
from tqdm.asyncio import tqdm


class MarkerLoader(BaseLoader):
    def __init__(self, page_sep: str = "[PAGE_SEP]", num_workers=2, **kwargs):
        super().__init__(**kwargs)
        self.page_sep = page_sep
        self.manager = MarkerManager(num_workers=num_workers, page_sep=page_sep)
        self._started = False

    async def start(self):
        if not self._started:
            await self.manager.start()
            self._started = True

    async def shutdown(self):
        if self._started:
            await self.manager.shutdown()
            self._started = False

    async def destroy(self):
        await self.shutdown()
        

    async def aload_document(self, file_path, metadata: dict = None, save_md=False):
        await self.start()

        logger.info(f"Loading {file_path}")
        start = time.time()

        render = await self.manager.submit_pdf(str(file_path))

        if isinstance(render, dict) and "error" in render:
            raise RuntimeError(f"Conversion failed: {render['error']}")

        text = render.markdown

        if self.config["loader"]["image_captioning"]:
            img_dict = render.images
            logger.info(f"Found {len(img_dict)} images in the document.")
            captions_dict = await self.get_captions(img_dict)
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



class MarkerWorker:
    def __init__(self, queue: asyncio.Queue, results: dict, page_sep: str):
        self.queue = queue
        self.results = results
        self.page_sep = page_sep

        self.converter = PdfConverter(
            artifact_dict=create_model_dict(),
            config={
                "output_format": "markdown",
                "paginate_output": True,
                "page_separator": self.page_sep,
            },
        )

    async def run(self):
        while True:
            task = await self.queue.get()
            if task is None:
                break  # poison pill to stop worker

            file_path, task_id = task
            try:
                render = self.converter(str(file_path))
                self.results[task_id] = render
            except Exception as e:
                self.results[task_id] = {"error": str(e)}
            finally:
                self.queue.task_done()


class MarkerManager:
    def __init__(self, num_workers: int, page_sep: str):
        self.queue = asyncio.Queue()
        self.results = {}
        self.page_sep = page_sep
        self.workers = [
            MarkerWorker(self.queue, self.results, page_sep)
            for _ in range(num_workers)
        ]
        self.worker_tasks = []

    async def start(self):
        self.worker_tasks = [asyncio.create_task(w.run()) for w in self.workers]

    async def shutdown(self):
        for _ in self.workers:
            await self.queue.put(None)  # poison pill
        await asyncio.gather(*self.worker_tasks)

    async def submit_pdf(self, file_path: str):
        task_id = str(uuid.uuid4())
        await self.queue.put((file_path, task_id))

        while task_id not in self.results:
            await asyncio.sleep(0.1)

        return self.results.pop(task_id)
