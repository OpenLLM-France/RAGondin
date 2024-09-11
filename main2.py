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
        answer, context = await ragPipe.run(question=question)
        async for chunk in answer:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")

        # await evaluate(ragPipe.llm_client, question, context)
        print("\n")

asyncio.run(main())