import asyncio
from src.components import RagPipeline, Config, evaluate
import time

config = Config()

start = time.time()
ragPipe = RagPipeline(config=config)
end = time.time()
print(f"Start Time: {end - start} s.")

# TODO: Talk about evaluation with metrics with Andrej

async def main():
    while True:
        question = input("Question sur vos documents: ")
        answer, context = ragPipe.run(question=question, chat_history=[])
        async for chunk in answer:
            print(chunk.content, end="")

        # print("\n")
        # await evaluate(ragPipe.llm_client, question, context)
        print("\n")

asyncio.run(main())