import chainlit as cl
import sys, os
from cachetools import TTLCache
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.components import RagPipeline, Config


config = Config()
# cache = TTLCache(maxsize=1, ttl=3*60)  # TTL en secondes
# ragPipe = cache.get('ragPipe', RagPipeline(config=config))
ragPipe = RagPipeline(config=config)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Hygiène numérique",
            message="Comment adopter une bonne hygiène numérique. Présente le résultat en un tableau simplissime.",
            icon="./public/idea.svg",
    ),
    cl.Starter(
            label="CLOUD ACT vs RGPD",
            message="Les conséquences du CLOUD ACT sous forme de tableau.",
            icon="./public/danger-triangle.svg",
            ),
    cl.Starter(
            label="Digital labor",
            message='Définition et effets de la notion de "Digital Labor".',
            icon="./public/labor-man-labor.svg",
            ),
    ]


@cl.on_message
async def main(message: cl.Message):
    question = message.content

    msg = cl.Message(content="")
    await msg.send()

    stream, _ = ragPipe.run(question)

    answer_txt = ""
    async for token in stream:
        await msg.stream_token(token.content)
        answer_txt += token.content
    
    if ragPipe.rag_mode == "ChatBotRag":
            ragPipe.update_history(question, answer_txt)

    await msg.update()
    