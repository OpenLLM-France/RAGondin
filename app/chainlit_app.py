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
            message="Quel est l'objectif d'OpenLLM France et les enjeux de ce projet?",
            icon="./public/idea.svg",
        ),

        cl.Starter(
                label="Produits de Linagora",
                message="Parle moi du Pack Twake et de LinTo?",
                icon="./public/labor-man-labor.svg",
                ),

        cl.Starter(
                label="Présentation de Linagora",
                message="Fais moi une présentation de Linagora.",
                icon= "./public/danger-triangle.svg",
                )
    ]


@cl.on_chat_start
async def on_chat_start():
    ragPipe._chat_history.clear()
    logger.info("Chat history flushed")
     

def format_elements(sources, only_txt=True):
    elements = []
    source_names = []
    for doc in sources:
        source_names.append(doc['doc_id'])

        if only_txt:
            elem = cl.Text(content=doc["content"], name=doc['doc_id'], display='side')
        else:
            source = Path(doc["source"])
            match source.suffix:
                case '.pdf':
                    elem = cl.Pdf(name=doc['doc_id'], path=doc["source"], page=doc["page"], display="side")

                case '.mp4':
                    elem = cl.Video(name=doc['doc_id'], path=doc["source"], display='side')
                
                case '.mp3':
                    elem = cl.Audio(name=doc['doc_id'], path=doc["source"], display='side')
                
                case _:
                    elem = cl.Text(content=doc["content"], name=doc['doc_id'], display='side') # TODO Maybe HTML (convert the File first)

        elements.append(elem)                
    return elements, source_names
     
    
@cl.on_message
async def main(message: cl.Message):
    question = message.content

    async with cl.Step(name="Searching for relevant documents...") as step:
        stream, _, sources = ragPipe.run(question)
    await step.remove()

    elements, source_names = format_elements(sources, only_txt=False)
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


if __name__ == "__main__":
    import sys
    from chainlit.cli import run_chainlit

    sys.argv.extend(["-w", "--no-cache"])
    run_chainlit(__file__)