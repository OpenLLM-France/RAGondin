#!/usr/bin/env python

import argparse
import asyncio
import os
import time
from components import Indexer, load_config
from loguru import logger


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

    if args.folder:
        indexer = Indexer(config, logger)

        start = time.time()
        await indexer.add_files2vdb(
            path=args.folder, partition=indexer.default_partition
        )
        end = time.time()

    if args.list:
        indexer = Indexer(config, logger)

        start = time.time()
        await indexer.add_files2vdb(path=args.list, partition=None)
        end = time.time()

    logger.info(f"Execution time: {end - start:.4f} seconds")
    if config.vectordb["enable"]:
        logger.info(
            f"Documents loaded to partition named '{indexer.default_partition}'."
        )


if __name__ == "__main__":
    asyncio.run(main())
