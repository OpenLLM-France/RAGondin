from pathlib import Path
from urllib.parse import urlparse, quote
import chainlit as cl
import yaml, torch
from loguru import logger
from io import BytesIO
import json
import httpx


BASE_URL = "http://localhost:8082/{method}/"
APP_DIR = Path.cwd().absolute()

headers = {
    "accept": "application/json", 
    "Content-Type": "application/json"
}

history = []

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
        url = quote(doc['url'], safe=':/')
        parsed_url = urlparse(doc['url'])
        doc_name = parsed_url.path.split('/')[-1]
        
        if only_txt:
            elem = cl.Text(content=doc["content"], name=doc['doc_id'], display='side')
            
        else:
            source = Path(url)
            match source.suffix:
                case '.pdf':
                    elem = cl.Pdf(name=doc['doc_id'], url=url, page=doc["page"], display='side')
                case '.mp4':
                    elem = cl.Video(name=doc['doc_id'], url=url, display='side')                
                case '.mp3':
                    elem = cl.Audio(name=doc['doc_id'], url=url, display='side')                
                case _:
                    elem = cl.Text(content=doc["content"], name=doc['doc_id'], display='side', url=url) # TODO Maybe HTML (convert the File first)

        s = f"{doc['doc_id']}: {doc_name}"
        elements.append(elem) 
        source_names.append(s)               
    return elements, source_names


@cl.on_chat_start
async def on_chat_start():
    try:
        global history
        history.clear()
        logger.info("New Chat Started")
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.get(url=BASE_URL.format(method='hello'))
            print(response.text)
    except:
        logger.warning("Make sur the fastapi is up!!")


@cl.set_chat_profiles
async def chat_profiles():
    with open(APP_DIR / 'public' / 'conversation_starters.yaml') as file: # Load the YAML file
        data = yaml.safe_load(file)
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        response = await client.get(url=BASE_URL.format(method='collections'))
        collections = json.loads(response.text)
    
    return [
        cl.ChatProfile(
            **profile,
        ) 
        for profile in data['chat_profiles'] if profile['name'] in collections
    ]

    
@cl.on_message
async def on_message(message: cl.Message):
    global history
    user_message = message.content
    params = {
            "new_user_input": user_message
        }
    async with cl.Step(name="Searching for relevant documents...") as step:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            async with client.stream(
                'POST',
                BASE_URL.format(method='generate'), 
                params=params, 
                headers=headers,
                json=history
            ) as streaming_response:
                metadata_sources = streaming_response.headers.get("X-Metadata-Sources")
                sources = json.loads(metadata_sources)

                if sources:
                    elements, source_names = format_elements(sources, only_txt=False)
                    msg = cl.Message(content="", elements=elements)
                else:
                    msg = cl.Message(content="")
                
                await msg.send()
                answer_txt = ""
                async for token in streaming_response.aiter_bytes():
                    token = token.decode()
                    await msg.stream_token(token)
                    answer_txt += token
    
    history.extend([
        {'role': 'user', 'content': user_message},
        {'role': 'user', 'content': answer_txt}
    ])

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
