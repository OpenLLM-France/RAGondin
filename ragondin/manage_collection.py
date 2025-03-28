#!/usr/bin/env python

import argparse
import asyncio
import os
import time
from components import Indexer, load_config
from loguru import logger
from components.indexer.loaders.loader import get_files, DocSerializer


def is_valid_directory(path):
    if os.path.isdir(path):
        return os.path.abspath(path)
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory.")


async def main():
    parser = argparse.ArgumentParser(
        description="Adds local files from a folder to qdrant vector db"
    )
    # Add a folder argument for uploading to qdrant
    parser.add_argument(
        "-f", "--folder", type=is_valid_directory, help="Path to the folder"
    )

    # Add an override argument for Hydra config
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        help="Overrides for the Hydra configuration (e.g., vectordb.enable=true). Can be used multiple times.",
        default=None,
    )

    # Add a list of files argument for uploading to qdrant
    parser.add_argument("-l", "--list", type=str, nargs="+", help="List of file paths")

    args = parser.parse_args()

    # Load the config with potential overrides
    config = load_config(overrides=args.override)
    serializer = DocSerializer(data_dir=config.paths.data_dir, config=config)

    if args.folder:
        indexer = Indexer.remote(config, logger)
        chunk_tasks = []
        async for file_path in get_files(
            serializer.loader_classes, args.folder, recursive=True
        ):
            chunk_tasks.append(
                indexer.add_file.remote(path=file_path, metadata={}, partition=None)
            )

        # Await all tasks concurrently
        start = time.time()
        await asyncio.gather(*chunk_tasks)
        end = time.time()

    if args.list:
        indexer = Indexer.remote(config, logger)
        start = time.time()
        await indexer.add_files.remote(path=args.list)
        end = time.time()

    logger.info(f"Execution time: {end - start:.4f} seconds")
    if config.vectordb["enable"]:
        logger.info("Documents loaded to `default` partition.")


if __name__ == "__main__":
    asyncio.run(main())
