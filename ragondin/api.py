import asyncio
from typing import Annotated
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request, Depends
from enum import Enum
import json
import os
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
import uvicorn
from typing import Literal
from pathlib import Path
from components import RagPipeline, Indexer
from config import load_config
from loguru import logger
from chainlit.utils import mount_chainlit

APP_DIR = Path.cwd()
DATA_DIR = APP_DIR / 'data'
# Directory to store uploaded PDFs
UPLOAD_DIR = APP_DIR / 'data' / "upload_dir"
os.makedirs(UPLOAD_DIR, exist_ok=True)

config = load_config()

print("config.vectordb['host']", config.vectordb["host"])
print("config.vectordb['port']", config.vectordb["port"])

indexer = Indexer(config, logger)
ragPipe = RagPipeline(config=config, vectordb=indexer.vectordb, logger=logger)

class Tags(Enum):
    VDB = "VectorDB operations"
    LLM = "LLM Calls"

class ChatMsg(BaseModel):
    role: Literal["user", "assistant"]
    content: str

mapping = {
    "user": HumanMessage, 
    "assistant": AIMessage
}

app = FastAPI()
app.mount('/static', StaticFiles(directory=DATA_DIR.absolute(), check_dir=True), name='static')


def static_base_url_dependency(request: Request) -> str:
    return f"{request.url.scheme}://{request.client.host}:{request.url.port}/static"


def source2url(s: dict, static_base_url: str):
    s['url'] = f"{static_base_url}/{s['sub_url_path']}"
    s.pop("source")
    s.pop('sub_url_path')
    return s


@app.get("/collections/",
          summary="Get existant collections",
          tags=[Tags.VDB]
          )
async def get_collections() -> list[str]:
    return await indexer.vectordb.get_collections()


@app.post("/generate/",
          summary="Given a question, this endpoint allows to generate an answer grounded on the documents in the VectorDB",
          tags=[Tags.LLM]
          )
async def get_answer(
    new_user_input: str, chat_history: list[ChatMsg]=None,
    static_base_url: str = Depends(static_base_url_dependency)
    ):

    msgs: list[HumanMessage | AIMessage] = None
    if chat_history:
        msgs = [mapping[chat_msg.role](content=chat_msg.content) for chat_msg in chat_history]
    answer_stream, context, sources = await ragPipe.run(question=new_user_input, chat_history=msgs)
    
    sources = list(map(lambda x: source2url(x, static_base_url), sources))
    src_json = json.dumps(sources)

    async def send_chunk():
        async for token in answer_stream:
            yield token.content
          
    return StreamingResponse(
        send_chunk(), 
        media_type="text/event-stream",
        headers={"X-Metadata-Sources": src_json},
    )

@app.get("/heath_check/", summary="Toy endpoint to check that the api is up")
async def heath_check():
    return "RAG API is up."


mount_chainlit(app, './chainlit/app_front.py', path="/chainlit") # mount the default front


if __name__ == "__main__":
    uvicorn.run('api:app', host="0.0.0.0", port=8083, reload=True, proxy_headers=True) # 8083

# uvicorn api:app --reload --port 8083 --host 0.0.0.0

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # await ragPipe.indexer.add_files2vdb(UPLOAD_DIR)
#     yield
#     # Clean up the db
#     ragPipe.indexer.connector.client.delete_collection(
#         collection_name=config.vectordb["collection_name"]
#     )

# app = FastAPI(
#     # lifespan=lifespan
# )



# async def process_data(file: UploadFile):
#     mime_type = file.content_type
#     if mime_type not in ['application/pdf']:
#         raise HTTPException(
#             status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
#             detail=f"Unsupported file type: {mime_type}"
#         )
#     try:
#         file_path = UPLOAD_DIR / file.filename
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
            
#         # await ragPipe.indexer.add_file2vdb(file_path)
#         await ragPipe.indexer.add_files2vdb(file_path)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         pass
#         # Delete the temporary file after processing
#         # if os.path.exists(file.filename):
#         #     os.remove(file.filename)


# @app.post("/files_to_db/",
#           summary="Add file(s) to the Quandrant VDB",
#           tags=[Tags.VDB]
#           )
# async def upload_files(
#     files: Annotated[
#         list[UploadFile], 
#         File(
#             title="PDF Files", 
#             description="PDF files to add to the vector database"
#         )], 
#     ):
#     tasks = [process_data(file) for file in files]
#     await asyncio.gather(*tasks)