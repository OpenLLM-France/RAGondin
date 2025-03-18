import asyncio
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

from langchain_core.documents.base import Document

from ..utils import SingletonMeta
from .chunker import ABCChunker, ChunkerFactory, SemanticSplitter
from .embeddings import HFEmbedder
from .loaders.loader import DocSerializer
from .vectordb import ConnectorFactory


class Indexer(metaclass=SingletonMeta):
    """This class bridges static files with the vector store database."""

    def __init__(self, config, logger, device=None) -> None:
        """
        Initializes the Indexer class with the given configuration, logger, and optional device.

        Args:
            config (Config): Configuration object containing settings for the embedder, paths, llm, and insertion.
            logger (Logger): Logger object for logging information.
            device (str, optional): Device to be used by the embedder. Defaults to None.

        Attributes:
            serializer (DocSerializer): Serializer for document data.
            chunker (ABCChunker): Chunker object created using the ChunkerFactory.
            vectordb (VectorDB): Vector database connector created using the ConnectorFactory.
            logger (Logger): Logger object for logging information.
            n_concurrent_loading (int): Number of concurrent loading operations. Defaults to 2.
            n_concurrent_chunking (int): Number of concurrent chunking operations. Defaults to 2.
        """
        embedder = HFEmbedder(embedder_config=config.embedder, device=device)
        self.serializer = DocSerializer(data_dir=config.paths.data_dir, config=config)
        self.chunker: ABCChunker = ChunkerFactory.create_chunker(
            config, embedder=embedder.get_embeddings()
        )
        self.vectordb = ConnectorFactory.create_vdb(
            config, logger=logger, embeddings=embedder.get_embeddings()
        )
        self.logger = logger
        self.logger.info("Indexer initialized...")
        self.config = config

        self.n_concurrent_loading = config.insertion.get(
            "n_concurrent_loading", 2
        )  # Number of concurrent loading operations
        self.n_concurrent_chunking = config.insertion.get(
            "n_concurrent_chunking", 2
        )  # Number of concurrent chunking operations
        self.default_partition = "_default"
        self.enable_insertion = self.config.vectordb["enable"]

    async def chunk(
        self, doc, gpu_semaphore: asyncio.Semaphore
    ) -> AsyncGenerator[Document, None]:
        """
        Asynchronously chunks a document using the specified chunker.

        If the chunker is an instance of `SemanticSplitter`, it will use a GPU semaphore
        to manage access to GPU resources while splitting the document.

        Args:
            doc (Document): The document to be chunked.
            gpu_semaphore (asyncio.Semaphore): A semaphore to control access to GPU resources.

        Returns:
            AsyncGenerator[Document, None]: An asynchronous generator yielding the chunks of the document.
        """
        if isinstance(self.chunker, SemanticSplitter):
            async with gpu_semaphore:
                chunks = await self.chunker.split_document(doc)
        else:
            chunks = await self.chunker.split_document(doc)

        self.logger.info(f"{Path(doc.metadata['source']).name} CHUNKED")
        return chunks

    async def add_files2vdb(
        self,
        path: str | list[str],
        metadata: Optional[Dict] = {},
        partition: Optional[str] = None,
    ):
        """
        Add files to the vector database in async mode.
        This method serializes documents from the given path(s) and processes them in chunks using GPU operations.
        The processed chunks are then added to the vector database.
        Args:
            path (str | list[str]): The path or list of paths to the files to be added.
            metadata (Optional[Dict], optional): Metadata to be associated with the documents. Defaults to an empty dictionary.
            collection_name (Optional[str], optional): The name of the collection in the vector database. Defaults to None.
        Raises:
            Exception: If an error occurs during the document processing or adding to the vector database.
        Returns:
            None
        """
        partition = self._check_partition_str(partition)
        gpu_semaphore = asyncio.Semaphore(
            self.n_concurrent_chunking
        )  # Only allow max_concurrent_gpu_ops GPU operation at a time
        doc_generator: AsyncGenerator[Document, None] = (
            self.serializer.serialize_documents(
                path,
                metadata={**metadata, "partition": partition},
                recursive=True,
            )
        )

        chunk_tasks = []
        try:
            async for doc in doc_generator:
                task = asyncio.create_task(self.chunk(doc, gpu_semaphore))
                chunk_tasks.append(task)

            # Await all tasks concurrently
            results = await asyncio.gather(*chunk_tasks)
            all_chunks = sum(results, [])

            if all_chunks:
                if self.enable_insertion:
                    await self.vectordb.async_add_documents(all_chunks)
                    self.logger.debug(f"Documents {path} added.")
                else:
                    self.logger.debug(
                        f"Documents {path} handled but not added to the database."
                    )

        except Exception as e:
            self.logger.error(f"An exception as occured: {e}")
            raise Exception(f"An exception as occured: {e}")

    def delete_file(self, file_id: str, partition: str):
        """
        Deletes files from the vector database based on the provided filters.
        Args:
            filters (Union[Dict, List[Dict]]): A dictionary or list of dictionaries containing the filters to identify files to be deleted.
            collection_name (Optional[str]): The name of the collection from which files should be deleted. Defaults to None.
        Returns:
            Tuple[List[Dict], List[Dict]]: A tuple containing two lists:
                - deleted_files: A list of filters for the files that were successfully deleted.
                - not_found_files: A list of filters for the files that were not found in the database.
        Raises:
            Exception: If an error occurs during the deletion process, it is logged and the function continues with the next filter.
        """
        if not self.enable_insertion:
            self.logger.error(
                "Vector database is not enabled, however, the delete_files method was called."
            )
            return

        try:
            # Get points associated with the file name
            points = self.vectordb.get_file_points(file_id, partition)
            if not points:
                self.logger.info(f"No points found for file_id: {file_id}")
                return
            # Delete the points
            self.vectordb.delete_points(points)
            self.logger.info(f"File {file_id} deleted.")
        except Exception as e:
            self.logger.error(f"Error in `delete_files` for file_id {file_id}: {e}")
            raise

        return True

    async def update_file_metadata(self, file_id: str, metadata: dict, partition: str):
        """
        Updates the metadata of a file in the vector database.
        Args:
            file_id (str): The ID of the file to be updated.
            metadata (Dict): The new metadata to be associated with the file.
            collection_name (Optional[str]): The name of the collection in which the file is stored. Defaults to None.
        Returns:
            None
        """
        if not self.enable_insertion:
            self.logger.error(
                "Vector database is not enabled, however, the update_file_metadata method was called."
            )
            return

        try:
            # Get existing chunks associated with the file name
            docs = self.vectordb.get_file_chunks(file_id, partition)

            # Update the metadata
            for doc in docs:
                doc.metadata.update(metadata)

            # Delete the existing chunks
            self.delete_file(file_id, partition)

            # Add the updated chunks
            if self.enable_insertion:
                await self.vectordb.async_add_documents(docs)

            self.logger.info(f"Metadata for file {file_id} updated.")
        except Exception as e:
            self.logger.error(
                f"Error in `update_file_metadata` for file_id {file_id}: {e}"
            )
            raise

    async def asearch(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: int = 0.80,
        partition: Optional[str | List[str]] = None,
        filter: Optional[Dict] = {},
    ) -> List[Document]:
        partition = self._check_partition_list(partition)
        results = await self.vectordb.async_search(
            query=query,
            partition=partition,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter=filter,
        )
        return results

    def _check_partition_str(self, partition: Optional[str]):
        if partition is None:
            self.logger.warning("Partition not provided. Using default partition.")
            partition = self.default_partition
        elif not isinstance(partition, str):
            raise ValueError("Partition should be a string.")
        return partition

    def _check_partition_list(self, partition: Optional[str]):
        if partition is None:
            self.logger.warning("Partition not provided. Using default partition.")
            partition = [self.default_partition]
        elif isinstance(partition, str):
            partition = [partition]
        elif (not isinstance(partition, list)) or (
            not all(isinstance(p, str) for p in partition)
        ):
            raise ValueError("Partition should be a string or a list of strings.")
        return partition
