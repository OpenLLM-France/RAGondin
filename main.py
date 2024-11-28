import time 
import asyncio
from loguru import logger
from pathlib import Path
from src.components import RagPipeline, load_config, evaluate
from filecatcher.components import Indexer

# config_path = Path(__file__) / '.hydra_config'
config = load_config()
indexer = Indexer(config, logger)    


async def main():
    start = time.time()
    # await indexer.add_files2vdb(path='./app/upload_dir/S2_RAG/Sources RAG/MARAP/PR_Memoire_Technique_VdM_18182_V8.odt')
    await indexer.add_files2vdb(path='./app/upload_dir/S2_RAG/Sources RAG/AI')
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