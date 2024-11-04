from abc import abstractmethod, ABC
import asyncio
import itertools
import os
from collections import defaultdict
import gc
import whisperx
from .utils import SingletonMeta
from pydub import AudioSegment
import torch
import math
from pathlib import Path
from typing import AsyncGenerator
from langchain_core.documents.base import Document
from langchain_community.document_loaders import (
    PyMuPDFLoader, 
    UnstructuredODTLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)
# from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
import pymupdf4llm
from loguru import logger
from tqdm.asyncio import tqdm  
from tqdm.asyncio import tqdm_asyncio # For async progress bars


from langchain_community.document_loaders import UnstructuredXMLLoader, PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader


class BaseLoader(ABC):
    @abstractmethod
    async def aload_doc(self, file_path):
        pass

    async def load(self, file_paths: list[Path | str]): 
        for fp in file_paths:
            yield await self.aload_doc(fp)


class Custompymupdf4llm(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', **loader_args) -> None:
        self.loader_args = loader_args
        self.page_sep = page_sep
    
    async def aload_doc(self, file_path):
        pages = pymupdf4llm.to_markdown(
            file_path,
            write_images=False,
            page_chunks=True,
            **self.loader_args
        )
        page_content = f'{self.page_sep}'.join([p['text'] for p in pages])
        return Document(
            page_content=page_content, 
            metadata={
                'source': pages[0]["metadata"]['file_path'],
                'page_sep': self.page_sep
            }
        )
    

class AudioTranscriber(metaclass=SingletonMeta):
    def __init__(self, device='cpu', compute_type='float32', model_name='large-v2', language='fr'):
        self.model = whisperx.load_model(
            model_name, 
            device=device, language=language, 
            compute_type=compute_type
        )
        
class VideoAudioLoader(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', batch_size=4):    
        self.batch_size = batch_size
        self.page_sep = page_sep
        self.formats = [".wav", '.mp3', ".mp4"]

        self.transcriber = AudioTranscriber(
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    
    def free_memory(self):
        gc.collect()
        torch.cuda.empty_cache()


    async def aload_doc(self, file_path):
        path = Path(file_path)
        if path.suffix not in self.formats:
            logger.warning(
                f'This audio/video file ({path.suffix}) is not supported.'
                f'The format should be {self.formats}'
            )
            return None

        # load it in wave format
        if path.suffix == '.wav':
            audio_path_wav = path
        else:
            sound = AudioSegment.from_file(file=path, format=path.suffix[1:])
            audio_path_wav = path.with_suffix('.wav')
            # Export to wav format
            sound.export(
                audio_path_wav, format="wav", 
                parameters=["-ar", "16000", "-ac", "1", "-ab", "32k"]
            )
        
        audio = whisperx.load_audio(audio_path_wav)

        if path.suffix != '.wav':
            os.remove(audio_path_wav)

        model = self.transcriber.model
        transcription_l = model.transcribe(audio, batch_size=self.batch_size)
        content = ' '.join([tr['text'] for tr in transcription_l['segments']])

        self.free_memory()

        return Document(
            page_content=f"{content}{self.page_sep}",
            metadata={
                'source':str(file_path),
                'page_sep': self.page_sep
            }
        )

class CustomPPTLoader(BaseLoader):
    doc_loaders = {
        '.pptx': UnstructuredPowerPointLoader, 
        '.ppt': UnstructuredPowerPointLoader
    }

    cat2md = {
        'Title': '#',
        'NarrativeText': '>',
        'ListItem': '-',
        'UncategorizedText': '',
        'EmailAddress': '',
        'FigureCaption': '*'
    }
    def __init__(self, page_sep: str='[PAGE_SEP]', **loader_args) -> None:
        self.loader_args = loader_args
        self.page_sep = page_sep
    
    def group_by_hierarchy(self, docs: list[Document]):
        """Group related elements within each page by parent_id and category_depth"""
        grouped_elements = defaultdict(list)
        
        # Sort by depth to ensure proper hierarchy
        sorted_docs = sorted(docs, key=lambda x: x.metadata.get('category_depth', 0))
        
        for doc in sorted_docs:
            parent_id = doc.metadata.get('parent_id', None)
            if parent_id:
                grouped_elements[parent_id].append(doc)
            else:
                grouped_elements[doc.metadata.get('element_id')].append(doc)
        return grouped_elements
    
    async def aload_doc(self, file_path):
        path = Path(file_path)
        cls_loader = CustomPPTLoader.doc_loaders.get(path.suffix, None)

        if cls_loader is None:
            raise Exception(f"This loader only supports {CustomPPTLoader.doc_loaders.keys()} format")
        
        loader = cls_loader(
            file_path=str(file_path), 
            mode='elements',
            **self.loader_args
        )
        elements = await loader.aload()

        # Step 1: Group elements by page_number
        grouped_by_page = defaultdict(list)
        for doc in elements:
            page_number = doc.metadata.get('page_number')
            grouped_by_page[page_number].append(doc)
        
        # Final structure: dictionary of pages, each containing grouped elements
        final_grouped_structure = {}
        for page, docs in grouped_by_page.items():
            final_grouped_structure[page] = self.group_by_hierarchy(docs)

        content = ''
        for page, grouped_elements in final_grouped_structure.items():
            t = ''
            for v in grouped_elements.values():
                for elem in v:
                    meta = elem.metadata
                    cat_md = CustomPPTLoader.cat2md.get(meta.get("category"), '')
                    t += f"{cat_md} {elem.page_content}\n"

            if page is not None and content:
                t = f'{self.page_sep}' + t
            
            content += t

        return Document(
            page_content=content, 
            metadata={
                'source': elements[0].metadata['source'],
                'page_sep': self.page_sep
            }
        )

class CustomPyMuPDFLoader(BaseLoader):
    def __init__(self, page_sep: str='[PAGE_SEP]', **loader_args) -> None:
        self.loader_args = loader_args
        self.page_sep = page_sep

    async def aload_doc(self, file_path):
        loader = PyMuPDFLoader(
            file_path=file_path,
            **self.loader_args
        )
        pages = await loader.aload()
        return Document(
            page_content=f'{self.page_sep}'.join([p.page_content for p in pages]), 
            metadata={
                'source': pages[0].metadata['source'],
                'page_sep': self.page_sep
            }
        )


class CustomDocLoader(BaseLoader):
    doc_loaders = {
            ".docx": UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.odt': UnstructuredODTLoader
        }
    def __init__(self, page_sep: str='[PAGE_SEP]', **loader_args) -> None:
        self.loader_args = loader_args
        self.page_sep = page_sep
    
    
    async def aload_doc(self, file_path):
        path = Path(file_path)
        cls_loader = CustomDocLoader.doc_loaders.get(path.suffix, None)

        if cls_loader is None:
            raise f"This loader only supports {CustomDocLoader.doc_loaders.keys()} format"
        
        loader = cls_loader(
            file_path=str(file_path), 
            mode='single',
            **self.loader_args
        )

        pages = await loader.aload()

        content = ' '.join([p.page_content for p in pages])

        return Document(
            page_content=f"{content}{self.page_sep}", 
            metadata={
                'source': pages[0].metadata['source'],
                'page_sep': self.page_sep
            }
        )


class GeneralLoader:
    def __init__(self, recursive=True, batch_size=16) -> None:
        self.recursive = recursive
        self.bs = batch_size

    async def doc_generator(self, path) -> AsyncGenerator[list[Document], None]:
        p = Path(path)
        if p.is_file():
            doc_batch = []
            
            async for doc in load_documents([p]):  # async iteration
                doc_batch.append(doc)
            
            # TODO: doc_batch = await load_documents([p])
        
            yield doc_batch


        if p.is_dir():
            for file_type in LOADERS.keys(): # TODO Rendre ceci async: Priority 0
                pattern = f"**/*{file_type}"

                files = get_files(path, pattern, self.recursive)
                files2 = get_files(path, pattern, self.recursive)

                n_files = len(list(files2))
                n_batches = math.ceil(n_files / self.bs)
                logger.info(f'Loading {n_files} {file_type} files.')

                async for f_batch in tqdm_asyncio(get_batches(files, batch_size=self.bs), total=n_batches, desc=f"Loading {file_type} files..."):
                    doc_batch = []
                    async for doc in load_documents(f_batch):  # async iteration
                        doc_batch.append(doc)
                    
                    yield doc_batch  # yield individual batches

def get_files(path, pattern, recursive):

    p = Path(path)
    files = p.rglob(pattern) if recursive else p.glob(recursive)

    # TODO: rendre le generator async ==> Faire du batching à la volée
    # b = []
    # async for f in files:
    #     b.append(f)
    #     # if size(b) == bs:
    #     yield b

    return files


async def load_documents(paths: list[Path]):
    # Use Terms like Serialize / Unserialize
    file_type = paths[0].suffix
    loader: BaseLoader = LOADERS.get(file_type)()
    async for doc in loader.load(paths):  # async iteration
        yield doc  # yield each document asynchronously


async def get_batches(generator, batch_size=16):
    it = iter(generator)
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            break
        yield batch




LOADERS = {
    '.pdf': CustomPyMuPDFLoader,

    '.docx': CustomDocLoader,
    '.doc': CustomDocLoader,
    '.odt': CustomDocLoader,

    '.mp4': VideoAudioLoader,
    '.pptx': CustomPPTLoader,
}



if __name__ == "__main__":
    async def main():
        loader = GeneralLoader(recursive=True, batch_size=16)
        dir_path = "../../app/upload_dir/Sources_RAG"  # Replace with your actual directory path

        # Use the doc_generator method
        async for doc_batch in loader.doc_generator(dir_path):
            print(f"Document source {doc_batch[0]}")
            print("-" * 50)

    asyncio.run(main())
