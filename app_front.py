import json
from pathlib import Path
from urllib.parse import quote, urlparse
from openai import AsyncOpenAI

import chainlit as cl
import httpx
from loguru import logger


PARTITION = "all"
headers = {"accept": "application/json", "Content-Type": "application/json"}

history = []


def get_base_url():
    from chainlit.context import get_context

    referer = get_context().session.http_referer
    parsed_url = urlparse(referer)  # Parse the referer URL
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url


@cl.set_chat_profiles
async def chat_profile():
    # base_url = get_base_url()
    # logger.info(f"base url {base_url}")

    # client = AsyncOpenAI(base_url=base_url, api_key="sk-1234")

    # l = await client.models.list()
    # logger.info(f"List: {l}")

    return [
        cl.ChatProfile(
            name="GPT-3.5",
            markdown_description="The underlying LLM model is **GPT-3.5**.",
            icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="GPT-4",
            markdown_description="The underlying LLM model is **GPT-4**.",
            icon="https://picsum.photos/250",
        ),
    ]


# this file is in the docker along with the fastapi running at port 8080


def format_elements(sources, only_txt=True):
    elements = []
    source_names = []
    for doc in sources:
        url = quote(doc["url"], safe=":/")
        parsed_url = urlparse(doc["url"])
        doc_name = parsed_url.path.split("/")[-1]

        if only_txt:
            elem = cl.Text(content=doc["content"], name=doc["doc_id"], display="side")

        else:
            source = Path(url)
            # logger.debug(f"Source: {url}")
            match source.suffix:
                case ".pdf":
                    elem = cl.Pdf(
                        name=doc["doc_id"], url=url, page=doc["page"], display="side"
                    )
                case ".mp4":
                    elem = cl.Video(name=doc["doc_id"], url=url, display="side")
                case ".mp3":
                    elem = cl.Audio(name=doc["doc_id"], url=url, display="side")
                case _:
                    elem = cl.Text(
                        content=doc["content"],
                        name=doc["doc_id"],
                        display="side",
                        url=url,
                    )  # TODO Maybe HTML (convert the File first)

        s = f"{doc['doc_id']}: {doc_name} ({doc['page']})"
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
        async with httpx.AsyncClient(timeout=httpx.Timeout(4 * 60.0)) as client:
            response = await client.get(url=f"{base_url}/health_check")

            print(response.text)
    except Exception as e:
        logger.error(f"An error happened: {e}")
        logger.warning("Make sur the fastapi is up!!")
    cl.user_session.set("BASE URL", base_url)


@cl.on_message
async def on_message(message: cl.Message):
    base_url = get_base_url()
    user_message = message.content
    params = {"new_user_input": user_message}
    async with cl.Step(name="Searching for relevant documents..."):
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(4 * 60.0), http2=True
        ) as client:
            async with client.stream(
                "POST",
                f"{base_url}/{PARTITION}/generate",
                params=params,
                headers=headers,
                json=history,
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

    history.extend(
        [
            {"role": "user", "content": user_message},
            {"role": "user", "content": answer_txt},
        ]
    )

    if sources:
        await msg.stream_token(
            "\n\n" + "-" * 50 + "\n\nRetrieved Docs: \n" + "\n".join(source_names)
        )

    await msg.send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
