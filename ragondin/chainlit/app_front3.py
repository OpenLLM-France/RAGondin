import chainlit as cl
from openai import AsyncOpenAI


base_url = "http://163.114.159.151:8080/v1"
api_key = "sk-1234"

client = AsyncOpenAI(api_key=api_key, base_url=base_url)


@cl.on_message
async def on_message(message: cl.Message):
    response = await client.chat.completions.create(
        model="ragondin-all",
        temperature=0,
        messages=[{"content": message.content, "role": "user"}],
    )
    await cl.Message(content=response.choices[0].message.content).send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
