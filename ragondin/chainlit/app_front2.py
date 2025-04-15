import asyncio
import os
from urllib.parse import urlparse
import chainlit as cl
from loguru import logger
import httpx
from utils.dependencies import vectordb
from openai import AsyncOpenAI
from chainlit.context import get_context

PARTITION = "all"
headers = {"accept": "application/json", "Content-Type": "application/json"}
history = []

def get_base_url():
    try:
        context = get_context()
        referer = context.session.http_referer
        parsed_url = urlparse(referer)  # Parse the referer URL
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        return base_url
    except Exception as e:
        logger.error(f"Error retrieving Chainlit context: {e}")
        return "http://localhost:8080"  # Default fallback URL


@cl.on_chat_start
async def on_chat_start():
    base_url = get_base_url()
    logger.debug(f"BASE URL: {base_url}")

    try:
        global history
        history.clear()
        logger.debug("New Chat Started")

        partition_names = vectordb.list_partitions()

        partition_choice = await cl.AskActionMessage(
            content="Select a partition to use:",
            actions=[
                cl.Action(name=partition, payload={"value": partition}, label=partition)
                for partition in partition_names
            ],
        ).send()

        if partition_choice:
            partition_selector = partition_choice.get("name")
            logger.debug(f"Selected partition: {partition_selector}")
            cl.user_session.set("selected_partition", partition_selector)

        async with httpx.AsyncClient(timeout=httpx.Timeout(4 * 60.0)) as client:
            response = await client.get(url=f"{base_url}/health_check")
            print(response.text)

            # value = partition_selector['partition_selector']

    except Exception as e:
        logger.error(f"An error happened: {e}")
        logger.warning("Make sur the fastapi is up!!")
    cl.user_session.set("BASE URL", base_url)


@cl.on_message
async def on_message(message: cl.Message):
    base_url = get_base_url()
    user_message = message.content

    uploaded_files = message.elements
    # file_sending_url = "http://".join([os.environ.get("BASE_URL"),":",os.environ.get("APP2_PORT")])
    if uploaded_files:
        for uploaded_file in uploaded_files:
            logger.debug(f"Uploaded file: {uploaded_file.name}")
            file_path = uploaded_file.path
            filename = uploaded_file.name
            file_ext = filename.split(".")[-1]

            # Send the file to the FastAPI server
            partition_name = filename
            file_id = file_ext
            url_template = (
                f"{base_url}/indexer/partition/{partition_name}/file/{file_id}"
            )
            url = url_template.format(partition_name=partition_name, file_id=filename)

            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(10 * 60.0)
                ) as client:
                    response = await client.get(url=f"{base_url}/health_check")
                    logger.debug(f"Health Check Response: {response.text}")

                with open(file_path, "rb") as f:
                    files = {
                        "file": (filename, f, f"application/{file_ext}"),
                        "metadata": (None, ""),
                    }

                    response = httpx.post(
                        url, files=files, headers={"accept": "application/json"}
                    )

                if response.status_code == 200:
                    await cl.Message(content="File uploaded successfully!").send()

                    vectorization_status_url = f"{base_url}/indexer/partition/{partition_name}/file/{file_id}/status"
                    while True:
                        try:
                            async with httpx.AsyncClient(
                                timeout=httpx.Timeout(10 * 60.0)
                            ) as client:
                                status_response = await client.get(
                                    vectorization_status_url
                                )
                            if status_response.status_code == 200:
                                status_data = status_response.json()
                                logger.debug(
                                    f"Status data: {status_data.get('status')}"
                                )
                                if status_data.get("status") == "vectorized":
                                    logger.debug(
                                        f"File {filename} has been vectorized."
                                    )
                                    break
                                else:
                                    logger.debug(
                                        f"Waiting for file {filename} to be vectorized..."
                                    )
                                    await asyncio.sleep(1)
                            else:
                                logger.error(
                                    f"Failed to check vectorization status for {filename}: {status_response.text}"
                                )
                                await cl.Message(
                                    content=f"Error checking vectorization status for {filename}."
                                ).send()
                                break
                        except httpx.TimeoutException:
                            logger.warning(
                                f"Timeout while checking vectorization status for {filename}. Retrying..."
                            )
                            await asyncio.sleep(2)
                else:
                    await cl.Message(
                        content=f"Failed to upload file: {response.text}"
                    ).send()

            except httpx.TimeoutException:
                logger.error(f"Timeout occurred while uploading file: {filename}")
                await cl.Message(
                    content=f"Timeout occurred while uploading file: {filename}. Please try again."
                ).send()

            except Exception as e:
                logger.error(f"Error uploading file: {e}")
                await cl.Message(
                    content="An error occurred while uploading the file."
                ).send()

    # Partition Selection
    selected_partition = cl.user_session.get(
        "selected_partition", "all"
    )  # default fallback

    logger.debug(f"Selected partition:{selected_partition}")
    api_key = "sk-1234"  # os.environ.get('API_KEY')

    client_openai = AsyncOpenAI(api_key=api_key, base_url=f'{base_url}/v1', timeout=4 * 60, default_headers=headers)

    try:
        response = await client_openai.chat.completions.create(
            messages=[
                {"role": "user", "content": user_message},
            ],
            model= f"ragondin-{selected_partition}",
            temperature=0,
            top_p=1,
            n=1,
            max_tokens=200,
            stream=False,
            stop=None,
        )

        await cl.Message(content=response.choices[0].message.content).send()

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        await cl.Message(
            content="An error occurred while processing your request."
        ).send()


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
