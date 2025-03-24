import asyncio
import time

from components import Indexer, load_config
from loguru import logger

config = load_config()
indexer = Indexer(config, logger)

L = [
    "./data/S2_RAG/Sources RAG/AI/LINAGORA Presentation LinTO DGE.pdf",
    "./data/S2_RAG/Sources RAG/AI/PR_18219_MAIF_Expertise_AT_IA_LLMops.pdf",
]


async def main():
    start = time.time()
    # await indexer.add_files2vdb(path='./data/tuto/')
    await indexer.add_files("/app/data/test/IA Ethique.pdf")
    # ragPipe = RagPipeline(config=config)
    # # await ragPipe.indexer.add_files2vdb("./app/upload_dir")
    end = time.time()
    print(f"Start Time: {end - start} s.")


asyncio.run(main())
