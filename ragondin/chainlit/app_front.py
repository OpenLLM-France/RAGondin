from pathlib import Path
from urllib.parse import urlparse, quote
import chainlit as cl
import yaml
from loguru import logger
import json
import httpx

headers = {
    "accept": "application/json", 
    "Content-Type": "application/json"
}

PARTITION = "all/"

history = []

def get_base_url():
    from chainlit.context import get_context
    referer = get_context().session.http_referer
    parsed_url = urlparse(referer) # Parse the referer URL
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/" + "{partition}{method}/"
    return base_url


# @cl.set_starters
# async def set_starters():
#     with open(APP_DIR / 'public' / 'conversation_starters.yaml') as file: # Load the YAML file
#         data = yaml.safe_load(file)
        
#     return [
#         cl.Starter(
#             label=item["label"],
#             message=item["message"],
#             icon=item["icon"]
#         )
#         for item in data['starters']
#     ]
     

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
            logger.debug(f"Url: {url}")
            match source.suffix:
                case '.pdf':
                    elem = cl.Pdf(name=doc['doc_id'], url=url, page=doc["page"], display='side')
                case '.mp4':
                    elem = cl.Video(name=doc['doc_id'], url=url, display='side')                
                case '.mp3':
                    elem = cl.Audio(name=doc['doc_id'], url=url, display='side')                
                case _:
                    elem = cl.Text(content=doc["content"], name=doc['doc_id'], display='side', url=url) # TODO Maybe HTML (convert the File first)

        s = f"{doc['doc_id']}: {doc_name} ({doc["page"]})"
        elements.append(elem) 
        source_names.append(s)               
    return elements, source_names

@cl.on_chat_start
async def on_chat_start():
    base_url = get_base_url()
    logger.debug(f"BASE URL: {base_url}")

    try:
        global history
        history.clear()
        logger.debug("New Chat Started")
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.get(url=base_url.format(partition = "",method='health_check'))
            print(response.text)
    except Exception as e:
        logger.error(f"An error happened: {e}")
        logger.warning("Make sur the fastapi is up!!")
    cl.user_session.set("base_url", base_url)

@cl.on_message
async def on_message(message: cl.Message):
    base_url: str = get_base_url()
    user_message = message.content
    params = {
            "new_user_input": user_message
        }
    async with cl.Step(name="Searching for relevant documents...") as step:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0), http2=True) as client:
            async with client.stream(
                'POST',
                base_url.format(partition=PARTITION, method='generate'), 
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

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)


### Code for audio transcription that will be used later

# from io import BytesIO
# import torch
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