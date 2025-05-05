import asyncio
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from loguru import logger
from omegaconf import OmegaConf
from tqdm.asyncio import tqdm
from ..utils import llmSemaphore


class ABCChunker(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    async def split_document(self, doc: Document):
        pass


template = """**Objectif** : Rédiger succinctement un texte de contextualisation pour le fragment suivant en intégrant les éléments fournis.

**Consignes de rédaction** :
1. Prendre en compte :
   - **Source** : Métadonnées du document (CV, vidéos, propositions clients, etc.)
   - **Première page** : Structure/En-tête du document original
   - **Fragment précédent** : Contenu adjacent pour assurer la continuité

2. Contraintes :
   - Langue : Utiliser la langue du fragment actuel
   - Format de réponse : Texte brut uniquement (pas de titres/markdown)
   - Longueur : 1 phrase à 1 paragraphe selon la pertinence

**Contexte** :
- Source : {source}
- Première page :
{first_chunk}
- Fragment précédent :
{prev_chunk}

**Fragment à contextualiser** :
{chunk}
"""


class ChunkContextualizer:
    """ChunkContextualizer is a class that provides functionality to generate context for chunks of text using a language model.
    Attributes:
        contextual_retrieval (bool): Indicates whether contextual retrieval is enabled.
        context_generator (Optional[Callable]): A callable for generating context if contextual retrieval is enabled.
    Methods:
        __init__(contextual_retrieval: bool = False, llm: Optional[ChatOpenAI] = None):
            Initializes the ChunkContextualizer with optional contextual retrieval and language model.
        generate_context(first_page: str, prev_chunk: str, chunk: str, source: str, semaphore: asyncio.Semaphore = llmSemaphore) -> str:
        contextualize(chunks: list[Document], pages: list[str], source: str, n_concurrent_request: int = 5) -> list[str]:
    """

    def __init__(
        self, contextual_retrieval: bool = False, llm: Optional[ChatOpenAI] = None
    ):
        self.contextual_retrieval = contextual_retrieval
        self.context_generator = None
        if self.contextual_retrieval:
            assert isinstance(llm, ChatOpenAI), (
                "The `llm` should be of type `ChatOpenAI` if contextual_retrieval is `True`"
            )
            prompt = ChatPromptTemplate.from_template(template=template)
            self.context_generator = (prompt | llm | StrOutputParser()).with_retry(
                retry_if_exception_type=(Exception,),
                wait_exponential_jitter=False,
                stop_after_attempt=2,
            )

    async def generate_context(
        self,
        first_chunk: str,
        prev_chunk: str,
        chunk: str,
        source: str,
        semaphore: asyncio.Semaphore = llmSemaphore,
    ):
        """generates context for a given chunk of text."""
        async with semaphore:
            try:
                return await self.context_generator.ainvoke(
                    {
                        "first_chunk": first_chunk,
                        "prev_chunk": prev_chunk,
                        "chunk": chunk,
                        "source": source,
                    }
                )
            except Exception as e:
                logger.warning(
                    f"Error when contextualizing a chunk of this document `{source}`: {e}"
                )
                return ""

    async def contextualize(
        self,
        chunks: list[Document],
        pages: list[str],
        source: str,
    ) -> list[str]:
        """
        Contextualizes a list of document chunks by generating context for each chunk based on the previous chunk.
        Args:
            chunks (list[Document]): A list of Document objects representing the chunks to be contextualized.
            pages (list[str]): A list of strings representing the pages of the document.
            source (str): The source of the document.
            n_concurrent_request (int, optional): The number of concurrent requests to be made. Defaults to 5.
        Returns:
            list[str]: A list of strings where each string is the contextualized content of a chunk.
        Raises:
            Exception: If an error occurs during the contextualization process, it logs a warning with the error message.
        """
        if not self.contextual_retrieval or len(chunks) == 1:
            return [chunk.page_content for chunk in chunks]

        try:
            tasks = []
            for i in range(1, len(chunks)):
                prev_chunk = chunks[i - 1]
                curr_chunk = chunks[i]
                tasks.append(
                    self.generate_context(
                        first_chunk=chunks[0],
                        prev_chunk=prev_chunk.page_content,
                        chunk=curr_chunk.page_content,
                        source=source,
                    )
                )
            contexts = await tqdm.gather(
                *tasks,
                total=len(tasks),
                desc=f"Contextualizing chunks of *{Path(source).name}*",
            )
            chunk_format = "chunks' context: {chunk_context}\n\n=> chunk: \n{chunk}"
            return [
                chunk_format.format(chunk=chunk.page_content, chunk_context=task)
                for chunk, task in zip(chunks[1:], contexts)
            ]

        except Exception as e:
            logger.warning(f"Error when contextualizing chunks from `{source}`: {e}")


class RecursiveSplitter(ABCChunker):
    """RecursiveSplitter is a class that splits a document into chunks and contextualizes them."""

    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 20,
        contextual_retrieval: bool = True,
        llm: Optional[ChatOpenAI] = None,
        **args,
    ):
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda x: llm.get_num_tokens(x),
            **args,
        )
        self.contextualizer = ChunkContextualizer(
            contextual_retrieval=contextual_retrieval, llm=llm
        )

    async def split_document(self, doc: Document):
        text = ""
        page_idx = []
        metadata = doc.metadata
        source = metadata["source"]
        page_sep = doc.metadata["page_sep"]
        pages: list[str] = doc.page_content.split(sep=page_sep)
        start_index = 0

        for page_num, p in enumerate(pages, start=1):
            text += "\n" + p
            c = " ".join(self.splitter.split_text(text))
            end_index = len(c) - 1
            page_idx.append(
                {"start_idx": start_index, "end_idx": end_index, "page": page_num}
            )
            start_index = end_index

        # chunking the full text
        full_text_preprocessed = " ".join(self.splitter.split_text(text))

        # Split the full text into chunks
        filtered_chunks = []
        chunks = self.splitter.create_documents([text])
        chunks_w_context = await self.contextualizer.contextualize(
            [Document(page_content="")] + chunks,
            pages,
            source=source,
        )

        i = 0
        for chunk, chunk_w_context in zip(chunks, chunks_w_context):
            start_idx = full_text_preprocessed.find(chunk.page_content)
            while (
                not (page_idx[i]["start_idx"] <= start_idx <= page_idx[i]["end_idx"])
                and i < len(page_idx) - 1
            ):
                i += 1

            if len(chunk.page_content.strip()) > 1:
                chunk.page_content = chunk_w_context
                metadata.update({"page": page_idx[i]["page"]})
                chunk.metadata = dict(metadata)
                filtered_chunks.append(chunk)
        return filtered_chunks


class SemanticSplitter(ABCChunker):
    """SemanticSplitter is a class that splits a document into semantically meaningful chunks and contextualizes them."""

    def __init__(
        self,
        min_chunk_size: int = 1000,
        embeddings=None,
        breakpoint_threshold_amount=85,
        contextual_retrieval: bool = False,
        llm: Optional[ChatOpenAI] = None,
        **args,
    ) -> None:
        from langchain_experimental.text_splitter import SemanticChunker

        self.splitter = SemanticChunker(
            embeddings=embeddings,
            buffer_size=1,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            min_chunk_size=min_chunk_size,
            add_start_index=True,
            **args,
        )
        self.contextualizer = ChunkContextualizer(
            contextual_retrieval=contextual_retrieval, llm=llm
        )

    async def split_document(self, doc: Document):
        """Splits a document into chunks and contextualizes them."""
        text = ""
        page_idx = []
        metadata = doc.metadata
        source = metadata["source"]
        page_sep = metadata.pop("page_sep")
        pages = doc.page_content.split(sep=page_sep)

        start_index = 0
        for page_num, page_txt in enumerate(pages, start=1):
            text += "\n" + page_txt
            c = " ".join(re.split(self.splitter.sentence_split_regex, text))
            end_index = len(c) - 1
            page_idx.append(
                {"start_idx": start_index, "end_idx": end_index, "page": page_num}
            )
            start_index = end_index

        chunks = self.splitter.create_documents(
            [" ".join(re.split(self.splitter.sentence_split_regex, text))]
        )

        filtered_chunks = []
        chunks = [Document(page_content="")] + chunks
        chunks_w_context = await self.contextualizer.contextualize(
            [Document(page_content="")] + chunks,
            pages,
            source=source,
            n_concurrent_request=5,
        )

        i = 0
        for chunk, chunk_w_context in zip(chunks, chunks_w_context):
            start_idx = chunk.metadata["start_index"]
            while not (page_idx[i]["start_idx"] <= start_idx <= page_idx[i]["end_idx"]):
                i += 1

            if len(chunk.page_content.strip()) > 1:
                chunk.page_content = chunk_w_context
                metadata.update({"page": page_idx[i]["page"]})
                chunk.metadata = dict(metadata)

                filtered_chunks.append(chunk)

        return filtered_chunks


class ChunkerFactory:
    CHUNKERS = {
        "recursive_splitter": RecursiveSplitter,
        "semantic_splitter": SemanticSplitter,
    }

    @staticmethod
    def create_chunker(
        config: OmegaConf,
        embedder: Optional[HuggingFaceBgeEmbeddings | HuggingFaceEmbeddings] = None,
    ) -> ABCChunker:
        def create_chunker(
            config: OmegaConf,
            embedder: Optional[HuggingFaceBgeEmbeddings | HuggingFaceEmbeddings] = None,
        ) -> ABCChunker:
            """
            Create and initialize a chunker based on the provided configuration.
            Args:
                config (OmegaConf): Configuration object containing chunker parameters.
                embedder (Optional[HuggingFaceBgeEmbeddings | HuggingFaceEmbeddings]): Optional embedder to be used if the chunker type is 'semantic_splitter'.
            """

        # Extract parameters
        chunker_params = OmegaConf.to_container(config.chunker, resolve=True)
        name = chunker_params.pop("name")

        # Initialize and return the chunker
        chunker_class: ABCChunker = ChunkerFactory.CHUNKERS.get(name)
        if not chunker_class:
            raise ValueError(f"Chunker '{name}' is not recognized.")

        # Add embeddings if semantic splitter is selected
        if name == "semantic_splitter":
            if embedder is not None:
                chunker_params.update({"embeddings": embedder})
            else:
                raise AttributeError(
                    f"{name} type chunker requires the `embedder` parameter"
                )

        # Include contextual retrieval if specified
        chunker_params["llm"] = ChatOpenAI(**config.vlm)
        return chunker_class(**chunker_params)
