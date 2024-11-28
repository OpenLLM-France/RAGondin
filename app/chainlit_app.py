from pathlib import Path
import chainlit as cl
import sys, os, yaml, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.components import RagPipeline, load_config
from loguru import logger

APP_DIR = Path(__file__).parent.absolute() # Path.cwd().parent.absolute()
UPLOAD_DIR = APP_DIR / "upload_dir"
os.makedirs(UPLOAD_DIR, exist_ok=True)

config = load_config() # Config(APP_DIR.parent / "config.ini")
config.vectordb["host"] = os.getenv('host')
config.vectordb["port"] = os.getenv('port')
print("config.vectordb['host']", config.vectordb["host"])
print("config.vectordb['port']", config.vectordb["port"])
ragPipe = RagPipeline(config=config, device="cpu")

# https://github.com/Cinnamon/kotaemon/blob/main/libs/ktem/ktem/reasoning/prompt_optimization/suggest_followup_chat.py

@cl.set_starters
async def set_starters():
    with open(APP_DIR / 'public' / 'conversation_starters.yaml') as file: # Load the YAML file
        data = yaml.safe_load(file)
        
    return [
        cl.Starter(
            label=item["label"],
            message=item["message"],
            icon=item["icon"]
        )
        for item in data['starters']
    ]
     

def format_elements(sources, only_txt=True):
    elements = []
    source_names = []
    for doc in sources:
        if only_txt:
            elem = cl.Text(content=doc["content"], name=doc['doc_id'], display='side')
            s = f"{doc['doc_id']}: {Path(doc["source"]).name}"
        else:
            source = Path(doc["source"])
            match source.suffix:
                case '.pdf':
                    elem = cl.Pdf(name=doc['doc_id'], path=doc["source"], page=doc["page"], display='side')
                    s = f"{doc['doc_id']}: {Path(doc["source"]).name} (p. {doc["page"]})"
                case '.mp4':
                    elem = cl.Video(name=doc['doc_id'], path=doc["source"], display='side')
                    s = f"{doc['doc_id']}: {Path(doc["source"]).name}"
                
                case '.mp3':
                    elem = cl.Audio(name=doc['doc_id'], path=doc["source"], display='side')
                    s = f"{doc['doc_id']}: {Path(doc["source"]).name}"
                
                case _:
                    elem = cl.Text(content=doc["content"], name=doc['doc_id'], display='side') # TODO Maybe HTML (convert the File first)
                    s = f"{doc['doc_id']}: {Path(doc["source"]).name}"

        elements.append(elem) 
        source_names.append(s)               
    return elements, source_names


@cl.on_chat_start
async def on_chat_start():
    ragPipe._chat_history.clear()
    logger.info("Chat history flushed")


@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name='vdb95',
            markdown_description='The underlying vector database is **vdb95**',
            icon='"https://picsum.photos/200"'
        ),
        cl.ChatProfile(
            name='vdb2',
            markdown_description='The underlying vector database is **vdb95**',
            icon='"https://picsum.photos/200"'
        )
    ]

    
@cl.on_message
async def on_message(message: cl.Message):
    question = message.content
    async with cl.Step(name="Searching for relevant documents...") as step:
        stream, _, sources = await ragPipe.run(question)
    await step.remove()

    if sources:
        elements, source_names = format_elements(sources, only_txt=True)
        msg = cl.Message(content="", elements=elements)
    else:
        msg = cl.Message(content="")
    
    await msg.send()
    answer_txt = ""
    
    async for token in stream:
        await msg.stream_token(token.content)
        answer_txt += token.content
    
    if ragPipe.rag_mode == "ChatBotRag":
            ragPipe.update_history(question, answer_txt)

    if sources:
        await msg.stream_token( '\n\n' + '-'*50 + "\n\nRetrieved Docs: \n" + '\n'.join(source_names))
        
    await msg.send()
    torch.cuda.empty_cache()



# from pydub import AudioSegment
# import whisperx
# from chainlit.element import ElementBased
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# audio_transcriber = AudioTranscriber(device=device).model


# @cl.on_audio_chunk
# async def on_audio_chunk(chunk: cl.AudioChunk):
#  if chunk.isStart:
#      buffer = BytesIO()
#      buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
#      cl.user_session.set("audio_buffer", buffer)
#      cl.user_session.set("audio_mime_type", chunk.mimeType)

#  # Write the chunks to a buffer
#  cl.user_session.get("audio_buffer").write(chunk.data)

# @cl.on_audio_end
# async def on_audio_end(elements: list[ElementBased]):
#  audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
#  # audio_mime_type: str = cl.user_session.get("audio_mime_type")
#  audio_buffer.seek(0)  # Move the file pointer to the beginning

#  try:
#      sound = AudioSegment.from_file(audio_buffer)
#      sound.export(
#          "output.wav", format="wav", 
#          parameters=["-ar", "16000", "-ac", "1", "-ab", "32k"]
#      )
#      trans_res= audio_transcriber.transcribe(
#          audio=whisperx.load_audio('output.wav'), batch_size=8
#      )
#      transcription = ' '.join(s['text'] for s in trans_res["segments"])

#      await cl.Message(content=f"transcription: {transcription}").send()

#      await on_message(cl.Message(content=transcription))

#  except Exception as e:
#      await cl.Message(content=f"Error processing audio: {str(e)}").send()

# chainlit run chainlit_app.py --host 0.0.0.0 --port 8000 --root-path /chainlit


if __name__ == "__main__":
    import sys
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)