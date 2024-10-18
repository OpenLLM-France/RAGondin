from pathlib import Path
import chainlit as cl
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.components import RagPipeline, Config
from loguru import logger

APP_DIR = Path(__file__).parent.absolute()
UPLOAD_DIR = APP_DIR / "upload_dir"
os.makedirs(UPLOAD_DIR, exist_ok=True)

config = Config(APP_DIR.parent / "config.ini")
ragPipe = RagPipeline(config=config)


# https://github.com/Cinnamon/kotaemon/blob/main/libs/ktem/ktem/reasoning/prompt_optimization/suggest_followup_chat.py

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="OpenLLM France",
            message="Quel est l'objectif d'OpenLLM France et quels sont les enjeux?",
            icon="./public/idea.svg",
    ),

    cl.Starter(
            label="Les logiciels développés de Linagora",
            message="Quels sont produits logiciels développés par Linagora?",
            icon="./public/labor-man-labor.svg",
            ),

    cl.Starter(
            label="Présentation de Linagora",
            message="Fais moi une présentation de Linagora.",
            icon= "./public/danger-triangle.svg",
            ),
    ]


@cl.on_chat_start
async def on_chat_start():
    ragPipe._chat_history.clear()
    logger.info("Chat history flushed")
     

@cl.on_message
async def main(message: cl.Message):
    question = message.content

    async with cl.Step(name="Searching for relevant documents...") as step:
        stream, _, sources, docs = ragPipe.run(question)
    await step.remove()


    # elements = [
    #     cl.Pdf(name=ref, path=s, page=p, display="side", size="medium") 
    #     for ref, s, p in sources
    # ]

    elements = []
    source_names = []
    for n, doc in enumerate(docs, start=1):
        source_names.append(f'[doc_{n}]')
        elements.append(cl.Text(content=doc.page_content, name=f'[doc_{n}]', display='side'))
    
    msg = cl.Message(content="", elements=elements)
    await msg.send()
    
    answer_txt = ""
    async for token in stream:
        await msg.stream_token(token.content)
        answer_txt += token.content
    
    if ragPipe.rag_mode == "ChatBotRag":
            ragPipe.update_history(question, answer_txt)

    await msg.stream_token("\n\nRetreived Docs: " + ', '.join(source_names))
    await msg.send()