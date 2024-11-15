#!/usr/bin/env python

import argparse
import asyncio
import time
import os
from loguru import logger
from qdrant_client import QdrantClient
from src.components import load_config

config = load_config() # Config("./config.ini")

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

    # Add a collection argument for specifying where to upload
    parser.add_argument("-c", "--collection",
        type=str,
        help="Name of the collection to which data will be added. If not provided the one from config.ini (vectordb.collection_name) will be used", 
        default=config.vectordb["collection_name"]
    )

    # Add a collection deletion argument
    parser.add_argument("-d", "--delete",
        type=str,
        help="Name of the collection to delete"
    )

    args = parser.parse_args()

    if args.folder:
        from src.components import Indexer

        collection = args.collection
        config.vectordb["collection_name"] = collection

        logger.warning(f"Data will be upserted to the collection {config.vectordb["collection_name"]}")
        indexer = Indexer(config, logger)
        
        start = time.time()
        await indexer.add_files2vdb(path=args.folder)
        end = time.time()

        print(f"Execution time: {end - start:.4f} seconds")
        logger.info(f"Documents loaded to collection named '{collection}'. ")
    
    
    if collection_name := args.delete:
        client = QdrantClient(
            port=config.vectordb["port"],
            host=config.vectordb["host"]
        )
        if client.collection_exists(collection_name=collection_name):
            client.delete_collection(collection_name)
            logger.info(f"collection '{collection_name}' deleted")
        else:
            logger.info(f"This collection doesn't exist")
    
if __name__ == '__main__':
    asyncio.run(main())
