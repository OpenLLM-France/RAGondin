import asyncio
from src.components import RagPipeline, Config, evaluate
import time

config = Config()

start = time.time()
ragPipe = RagPipeline(config=config)
end = time.time()
print(f"Start Time: {end - start} s.")


async def main():
    # await ragPipe.docvdbPipe.add_file2vdb("./Développement-Personnel-pdf-1.pdf")
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