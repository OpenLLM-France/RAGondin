import asyncio
from src.components import RagPipeline, Config, evaluate, Indexer
import time 
from loguru import logger

config = Config("./config.ini")
indexer = Indexer(config, logger)    


async def main():
    start = time.time()
    await indexer.add_files2vdb(path='./app/upload_dir/Sources_RAG')

    ragPipe = RagPipeline(config=config)
    # await ragPipe.indexer.add_files2vdb("./app/upload_dir")
    end = time.time()
    print(f"Start Time: {end - start} s.")

    while True:
        question = input("Question sur vos documents: ")
        answer, context, _ = ragPipe.run(question=question)
        answer_txt = ""
        async for chunk in answer:
            print(chunk.content, end="")
            answer_txt += chunk.content
        
        if ragPipe.rag_mode == "ChatBotRag":
            ragPipe.update_history(question, answer_txt)

        print("\n")
        evaluate(ragPipe.llm_client.client, context, ragPipe._chat_history, question, answer_txt)
        print("\n")

asyncio.run(main())