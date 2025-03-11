#!/usr/bin/env python

import argparse
import asyncio
import time
import os
from loguru import logger
from qdrant_client import QdrantClient
from components import load_config, Indexer

config = load_config()

def is_valid_directory(path):
    """Custom validation to check if the directory exists."""
    if os.path.isdir(path):
        return os.path.abspath(path)
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory.")


async def main():
    parser = argparse.ArgumentParser(description='Adds local files from a folder to qdrant vector db')
    # Add a folder argument for uploading to qdrant
    parser.add_argument("-f", "--folder",
        type=is_valid_directory,
        help="Path to the folder"
    )

    # Add a collection deletion argument
    parser.add_argument("-d", "--delete",
        type=str,
        help="Name of the collection to delete"
    )

    # Add an override argument for Hydra config
    parser.add_argument(
        "-o", "--override",
        action="append",
        help="Overrides for the Hydra configuration (e.g., vectordb.collection_name='vdb95'). Can be used multiple times.",
        default=None,
    )

    # Add a list of files argument for uploading to qdrant
    parser.add_argument("-l", "--list",
        type=str,
        nargs='+',
        help="List of file paths"
    )

    args = parser.parse_args()

    # Load the config with potential overrides
    config = load_config(overrides=args.override)

    if args.folder:
        collection = config.vectordb["collection_name"]
        if config.vectordb["enable"]:
            logger.warning(f"Data will be upserted to the collection {collection}")

        indexer = Indexer(config, logger)
        
        start = time.time()
        await indexer.add_files2vdb(path=args.folder)
        end = time.time()
    
    if collection_name := args.delete:
        end = 0
        start = 0
        client = QdrantClient(
            port=config.vectordb["port"],
            host=config.vectordb["host"]
        )
        if client.collection_exists(collection_name=collection_name):
            client.delete_collection(collection_name)
            logger.info(f"collection '{collection_name}' deleted")
        else:
            logger.info(f"This collection doesn't exist")
            return
    

    if args.list:
        collection = config.vectordb["collection_name"]
        if config.vectordb["enable"]:
            logger.warning(f"Data will be upserted to the collection {collection}")

        indexer = Indexer(config, logger)
        
        start = time.time()
        await indexer.add_files2vdb(path=args.list)
        end = time.time()

    logger.info(f"Execution time: {end - start:.4f} seconds")
    if config["vectordb"]["enable"]:
        logger.info(f"Documents loaded to collection named '{collection}'.")


    
if __name__ == '__main__':
    asyncio.run(main())

# ./manage_collection.py -f app/upload_dir/S2_RAG/ -o vectordb.collection_name='vdb90' -o chunker.breakpoint_threshold_amount=90
