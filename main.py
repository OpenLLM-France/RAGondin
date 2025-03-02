import time 
import asyncio
from loguru import logger
from pathlib import Path
from src.components import RagPipeline, load_config, Indexer

config = load_config()
indexer = Indexer(config, logger)

L = [
    './data/S2_RAG/Sources RAG/AI/LINAGORA Presentation LinTO DGE.pdf',
    './data/S2_RAG/Sources RAG/AI/PR_18219_MAIF_Expertise_AT_IA_LLMops.pdf'
]

async def main():
    start = time.time()
    # await indexer.add_files2vdb(path='./data/tuto/')
    await indexer.add_files2vdb(L)
    # ragPipe = RagPipeline(config=config)
    # # await ragPipe.indexer.add_files2vdb("./app/upload_dir")
    end = time.time()
    print(f"Start Time: {end - start} s.")

    # while True:
    #     question = input("Question sur vos documents: ")
    #     answer, context, *_ = ragPipe.run(question=question)
    #     answer_txt = ""
    #     async for chunk in answer:
    #         print(chunk.content, end="")
    #         answer_txt += chunk.content
        
    #     if ragPipe.rag_mode == "ChatBotRag":
    #         ragPipe.update_history(question, answer_txt)

    #     print("\n")
    #     evaluate(ragPipe.llm_client.client, context, ragPipe._chat_history, question, answer_txt)
    #     print("\n")

asyncio.run(main())


# docker run -p 6333:6333 -p 6334:6334 --name db_test -d\
#     -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
#     qdrant/qdrant