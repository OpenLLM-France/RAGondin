from pathlib import Path
import chainlit as cl
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.components import RagPipeline, Config
config = Config()
ragPipe = RagPipeline(config=config)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="IA Générative",
            message="Quels sont les bénéfices de l’IA générative?",
            icon="./public/idea.svg",
    ),

    cl.Starter(
            label="CLOUD ACT vs RGPD",
            message="Comment le CLOUD ACT se conforme au RGPD?",
            icon="./public/danger-triangle.svg",
            ),

    cl.Starter(
            label="Digital labor",
            message='Définition et effets de la notion de "Digital Labor".',
            icon="./public/labor-man-labor.svg",
            ),
    ]


@cl.on_chat_start
async def on_chat_start():
    ragPipe._chat_history.clear()
     


@cl.on_message
async def main(message: cl.Message):
    question = message.content

    async with cl.Step(name="Searching for relevant documents...") as step:
        stream, _, sources = ragPipe.run(question)
    await step.remove()


    elements = [
        cl.Pdf(name=ref, path=s, page=p, display="side", size="medium") 
        for ref, s, p in sources
    ]
    msg = cl.Message(content="", elements=elements)
    await msg.send()
    
    answer_txt = ""
    async for token in stream:
        await msg.stream_token(token.content)
        answer_txt += token.content
    
    if ragPipe.rag_mode == "ChatBotRag":
            ragPipe.update_history(question, answer_txt)

    await msg.send()
