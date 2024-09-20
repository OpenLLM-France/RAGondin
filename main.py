import asyncio
from src.components import RagPipeline, Config
import time

config = Config()

start = time.time()
ragPipe = RagPipeline(config=config)
end = time.time()
print(f"Start Time: {end - start} s.")


async def main():
    while True:
        question = input("Question sur vos documents: ")
        answer, context = ragPipe.run(question=question)
        answer_txt = ""
        async for chunk in answer:
            print(chunk.content, end="")
            answer_txt += chunk.content
        
        if ragPipe.rag_mode == "ChatBotRag":
            ragPipe.update_history(question, answer_txt)


        # print("\n")
        # await evaluate(ragPipe.llm_client, question, context)
        print("\n")

asyncio.run(main())