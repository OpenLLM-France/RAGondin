import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider


@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            Select(
                id="vectordb",
                label="Qdrant DB",
                values=["vdb95", "vdb95_no_context"],
                initial_index=0,
            ),
            Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),

            Slider(
                id="Threshold",
                label="Similarity - Threshold",
                initial=0.65,
                min=0.5,
                max=1,
                step=0.1,
                description=''
            ),
            Slider(
                id="nb_docs_reranker",
                label="Reranker - N",
                initial=7,
                min=3,
                max=10,
                step=1,
                description="Nb documents to keep after reranker",
            ),
        ]
    ).send()


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)

