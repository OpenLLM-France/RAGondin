from pathlib import Path
from urllib.parse import urlparse, quote
import chainlit as cl
import yaml
from loguru import logger
import json
import httpx


headers = {"accept": "application/json", "Content-Type": "application/json"}

history = []

BASE_URL = "http://0.0.0.0:8080/{method}/"  # this file is in the docker along with the fastapi running at port 8080


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
        url = quote(doc["url"], safe=":/")
        parsed_url = urlparse(doc["url"])
        doc_name = parsed_url.path.split("/")[-1]

        if only_txt:
            elem = cl.Text(content=doc["content"], name=doc["doc_id"], display="side")

        else:
            source = Path(url)
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
    logger.debug(f"BASE URL: {BASE_URL}")

    try:
        global history
        history.clear()
        logger.debug("New Chat Started")
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.get(url=BASE_URL.format(method="health_check"))
            print(response.text)
    except Exception as e:
        logger.error(f"An error happened: {e}")
        logger.warning("Make sur the fastapi is up!!")
    cl.user_session.set("BASE URL", BASE_URL)


@cl.on_message
async def on_message(message: cl.Message):
    user_message = message.content
    params = {"new_user_input": user_message}
    async with cl.Step(name="Searching for relevant documents...") as step:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0), http2=True) as client:
            async with client.stream(
                "POST",
                BASE_URL.format(method="generate"),
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