import httpx
import chainlit as cl
from loguru import logger
from openai import AsyncOpenAI

# from chainlit.server import app as chainlit_app
# from starlette.middleware.base import BaseHTTPMiddleware
# from fastapi import Request


# class CaptureBaseURL(BaseHTTPMiddleware):
#     """Stash the computed base_url into Chainlit’s user_session on each HTTP request."""

#     async def dispatch(self, request: Request, call_next):
#         scheme = request.url.scheme
#         netloc = request.url.netloc
#         chainlit_app.state.base_url = f"{scheme}://{netloc}"
#         return await call_next(request)


# chainlit_app.add_middleware(CaptureBaseURL)


def get_api_base_url():
    # return chainlit_app.state.base_url
    return "http://localhost:8080"


# Instrument the OpenAI client
# cl.instrument_openai()

base_url = get_api_base_url()
client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="sk-1234")


@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    try:
        output = await client.models.list()
        models = output.data
        chat_profiles = []

        for i, m in enumerate(models, start=1):
            partition = m.id.split("ragondin-")[1]

            description_template = "You are interacting with the **{name}** LLM.\n" + (
                "The LLM's answers will be grounded on **all** partitions."
                if "all" in m.id
                else "The LLM's answers will be grounded only on the partition named **{partition}**."
            )

            chat_profiles.append(
                cl.ChatProfile(
                    name=m.id,
                    markdown_description=description_template.format(
                        name=m.id, partition=partition
                    ),
                    icon=f"https://picsum.photos/{250 + i}",
                )
            )
        return chat_profiles

    except Exception as e:
        await cl.Message(content=f"An error occured: {str(e)}").send()


@cl.on_chat_start
async def on_chat_start():
    base_url = get_api_base_url()
    logger.debug(f"BASE URL: {base_url}")

    chat_profile = cl.user_session.get("chat_profile")
    settings = {"model": chat_profile, "temperature": 0, "stream": True}

    cl.user_session.set("messages", [])

    logger.debug("New Chat Started")
    cl.user_session.set("settings", settings)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(4 * 60.0)) as client:
            response = await client.get(url=f"{base_url}/health_check")
            print(response.text)

    except Exception as e:
        logger.error(f"An error happened: {e}")
        logger.warning("Make sur the fastapi is up!!")
    cl.user_session.set("BASE URL", base_url)


@cl.on_message
async def on_message(message: cl.Message):
    settings = cl.user_session.get("settings")
    messages: list = cl.user_session.get("messages", [])
    messages.append({"role": "user", "content": message.content})

    base_url = get_api_base_url()
    client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="sk-1234")

    stream = await client.chat.completions.create(
        messages=messages, timeout=4 * 60, **settings
    )

    # STREAM Response
    msg = cl.Message(content="")
    await msg.send()

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    await msg.update()

    messages.append({"role": "assistant", "content": msg.content})
    cl.user_session.set("messages", messages)


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
