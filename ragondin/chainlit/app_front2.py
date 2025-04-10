import json
from pathlib import Path
from urllib.parse import quote, urlparse
import chainlit as cl
from loguru import logger
import httpx
from openai import AsyncOpenAI


# Instrument the OpenAI client
cl.instrument_openai()


def get_base_url():
    from chainlit.context import get_context

    referer = get_context().session.http_referer
    parsed_url = urlparse(referer)  # Parse the referer URL
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url


@cl.on_chat_start
async def on_chat_start():
    base_url = get_base_url()
    settings = {
        "model": "ragondin-all",
        "temperature": 0,
        "timeout": 4 * 60.0,
    }

    logger.debug(f"BASE URL: {base_url}")

    try:
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
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    settings = cl.user_session.get("settings")
    base_url = cl.user_session.get("BASE URL")

    client = AsyncOpenAI(api_key="sk-1234", base_url=f"{base_url}/v1", timeout=4 * 60.0)
    stream = await client.chat.completions.create(
        messages=message_history, stream=True, **settings
    )

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    message_history.append({"role": "assistant", "content": msg.content})

    await msg.update()
    cl.user_session.set("message_history", message_history)


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
