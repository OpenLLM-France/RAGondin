import json
from pathlib import Path
from urllib.parse import quote, urlparse
import chainlit as cl
from loguru import logger
import httpx
from openai import AsyncOpenAI
from openai import OpenAI


# Instrument the OpenAI client
cl.instrument_openai()
PARTITION = "all"
headers = {"accept": "application/json", "Content-Type": "application/json"}
history = []

def get_base_url():
    from chainlit.context import get_context

    referer = get_context().session.http_referer
    parsed_url = urlparse(referer)  # Parse the referer URL
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url

def get_settings():
    settings = {
        "model": "ragondin-all",
        "temperature": 0,
        "timeout": 4 * 60.0,
    }
    return settings

@cl.on_chat_start
async def on_chat_start():
    base_url = get_base_url()
    settings = get_settings()

    logger.debug(f"BASE URL: {base_url}")

    try:
        global history
        history.clear()
        cl.user_session.set("message_history", [])
        cl.user_session.set("settings", settings)
        logger.debug("New Chat Started")
        async with httpx.AsyncClient(timeout=httpx.Timeout(4 * 60.0)) as client:
            response = await client.get(url=f"{base_url}/health_check")
            print(response.text)

    except Exception as e:
        logger.error(f"An error happened: {e}")
        logger.warning("Make sur the fastapi is up!!")
    cl.user_session.set("BASE URL", base_url)


@cl.on_message
async def on_message(message: cl.Message):

    # client = AsyncOpenAI(api_key="sk-1234", base_url=f"{get_base_url()}/v1", timeout=4 * 60.0)
    # response = await client.chat.completions.create(
    #     messages=[
    #         {
    #             "content": message.content,
    #             "role": "user"
    #         }
    #     ],
    #     **get_settings()
    # )
    print(message.content)

    base_url = get_base_url()
    api_key = 'sk-1234'

    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": message.content},
        ],
        model=f"ragondin-{PARTITION}",
        temperature=0,
        top_p=1,
        n=1,
        max_tokens=200,
        stream=False,
        stop=None
    )
    
    await cl.Message(content=response.choices[0].message.content).send()



# this file is in the docker along with the fastapi running at port 8080

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


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
