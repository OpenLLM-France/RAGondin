import json
from pathlib import Path
import httpx
import chainlit as cl
from loguru import logger
from openai import AsyncOpenAI
import os
from urllib.parse import urlparse
from chainlit.context import get_context

headers = {"accept": "application/json", "Content-Type": "application/json"}


def get_base_url():
    try:
        context = get_context()
        referer = context.session.http_referer
        parsed_url = urlparse(referer)  # Parse the referer URL
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        return base_url
    except Exception as e:
        logger.error(f"Error retrieving Chainlit context: {e}")
        port = os.environ.get("CONTAINER_PORT", "8080")
        logger.info(f"PORT: {port}")
        return f"http://localhost:{port}"  # Default fallback URL


@cl.set_chat_profiles
async def chat_profile():
    base_url = get_base_url()
    client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="sk-1234")

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
    base_url = get_base_url()
    logger.debug(f"BASE URL: {base_url}")

    chat_profile = cl.user_session.get("chat_profile")
    settings = {
        "model": chat_profile,
        "temperature": 0,
        "stream": True,
        "max_tokens": 1000,
    }

    cl.user_session.set("messages", [])

    logger.debug("New Chat Started")
    cl.user_session.set("settings", settings)

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=httpx.Timeout(4 * 60.0)), headers=headers
        ) as client:
            response = await client.get(url=f"{base_url}/health_check", headers=headers)
            print(response.text)

    except Exception as e:
        logger.error(f"An error happened: {e}")
        logger.warning("Make sur the fastapi is up!!")
    cl.user_session.set("BASE URL", base_url)


async def __fetch_page_content(chunk_url):
    async with httpx.AsyncClient() as client:
        response = await client.get(chunk_url)
        response.raise_for_status()  # raises exception for 4xx/5xx responses
        data = response.json()
        return data.get("page_content", "")


async def __format_sources(metadata_sources, only_txt=False):
    elements = []
    source_names = []
    for s in metadata_sources:
        filename = Path(s["filename"])
        file_url = s["file_url"]
        logger.info(f"URL: {file_url}")
        doc_id = s["doc_id"]
        page = s["page"]
        if only_txt:
            chunk_content = await __fetch_page_content(chunk_url=s["chunk_url"])
            elem = cl.Text(content=chunk_content, name=doc_id, display="side")
        else:
            match filename.suffix:
                case ".pdf":
                    elem = cl.Pdf(
                        name=doc_id, url=file_url, page=int(s["page"]), display="side"
                    )
                case ".mp4":
                    elem = cl.Video(name=doc_id, url=file_url, display="side")
                case ".mp3":
                    elem = cl.Audio(name=doc_id, url=file_url, display="side")
                case _:
                    # logger.info(f"Link: {s['chunk_url']}")
                    chunk_content = await __fetch_page_content(chunk_url=s["chunk_url"])
                    elem = cl.Text(content=chunk_content, name=doc_id, display="side")

        elements.append(elem)
        source_names.append(f"{doc_id}: {filename} (page: {page})")

    return elements, source_names


@cl.on_message
async def on_message(message: cl.Message):
    settings = cl.user_session.get("settings")
    messages: list = cl.user_session.get("messages", [])
    messages.append({"role": "user", "content": message.content})

    base_url = get_base_url()
    client = AsyncOpenAI(base_url=f"{base_url}/v1", api_key="sk-1234")

    payload = {
        "messages": messages,
        **dict(settings),
    }

    async with cl.Step(name="Searching for relevant documents..."):
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=httpx.Timeout(4 * 60.0)), http2=True
        ) as client:
            async with client.stream(
                "POST",
                url=f"{base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                metadata_sources = json.loads(resp.headers.get("X-Metadata-Sources"))
                if metadata_sources:
                    elements, source_names = await __format_sources(metadata_sources)
                    msg = cl.Message(content="", elements=elements)
                else:
                    msg = cl.Message(content="")

                # STREAM Response
                await msg.send()

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    content = line.removeprefix("data: ").strip()
                    if content == "[DONE]":
                        break

                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError as e:
                        logger.error(f"Erreur JSON sur chunk : {e!r}")
                        continue

                    # parse JSON
                    data = json.loads(content)
                    token = data.get("choices", [{}])[0].get("delta", {}).get("content")
                    if token:
                        await msg.stream_token(token)

                await msg.update()
                messages.append({"role": "assistant", "content": msg.content})
                cl.user_session.set("messages", messages)

                # Show sources
                s = "\n\n" + "-" * 50 + "\n\nSources: \n" + "\n".join(source_names)
                await msg.stream_token(s)
                await msg.update()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
