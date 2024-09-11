import chainlit as cl
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.components import RagPipeline, Config

config = Config()
ragPipe = RagPipeline(config=config)


@cl.on_chat_start
def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a helpful assistant."}],
    )


# @cl.set_starters
# async def set_starters():
#     return [
#         cl.Starter(
#             label="Hygiène numérique",
#             message="Comment adopter une bonne hygiène numérique. Présente le résultat en un tableau simplissime.",
#             icon="./public/idea.svg",
#     ),
#     cl.Starter(
#             label="CLOUD ACT vs RGPD",
#             message="Les conséquences du CLOUD ACT sous forme de tableau.",
#             icon="./public/danger-triangle.svg",
#             ),
#     cl.Starter(
#             label="Digital labor",
#             message='Définition et effets de la notion de "Digital Labor".',
#             icon="/public/labor-man-labor.svg",
#             ),
#     ]



@cl.on_message
async def main(message: cl.Message):
    question = message.content

    msg = cl.Message(content="")
    await msg.send()

    stream, _ = await ragPipe.run(question)

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    # message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()
    