from openai import AsyncOpenAI
import chainlit as cl
from src.pipeline2 import RagPipeline
from src.config import Config


config = Config()
ragPipe = RagPipeline(config=config)

settings = {}


@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )


@cl.on_message
async def main(message: cl.Message):
    question = message.content

    msg = cl.Message(content="")
    await msg.send()

    stream = await ragPipe.run(question=question)

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    # message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()


import chainlit as cl

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Hygiène numérique",
            message="Comment adopter une bonne hygiène numérique. Présente le résultat en un tableau simplissime.",
            icon="/public/idea.svg",
    ),
    cl.Starter(
            label="CLOUD ACT vs RGPD",
            message="Les conséquences du CLOUD ACT sous forme de tableau.",
            icon="/public/learn.svg",
            ),
    cl.Starter(
            label="Digital labor",
            message='Définition et effets de la notion de "Digital Labor".',
            icon="/public/learn.svg",
            ),
    ]