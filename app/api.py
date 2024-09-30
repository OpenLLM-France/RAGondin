import asyncio
from typing import Annotated
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from enum import Enum
from chainlit.utils import mount_chainlit
import sys, os
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import uvicorn
from typing import Literal
import aiofiles


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.components import RagPipeline, Config


config = Config()
ragPipe = RagPipeline(config=config)


class Tags(Enum):
    VDB = "VectorDB operations"
    LLM = "LLM Calls"
  
class ChatMsg(BaseModel):
    role: Literal["human", "assistant"]
    message: str


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     yield
#     # Clean up the db
#     ragPipe.docvdbPipe.connector.client.delete_collection(
#         collection_name=config.collection_name
#     )


app = FastAPI(
    # lifespan=lifespan
)


async def process_data(file: UploadFile):
    mime_type = file.content_type
    if mime_type not in ['application/pdf']:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {mime_type}"
        )
    try:
        with open(file.filename, "wb") as f:
            f.write(await file.read())
            
        await ragPipe.docvdbPipe.add_file2vdb(file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        pass
        # Delete the temporary file after processing
        # if os.path.exists(file.filename):
        #     os.remove(file.filename)



@app.post("/files_to_db/",
          summary="Add file(s) to the Quandrant VDB",
          tags=[Tags.VDB]
          )
async def upload_files(
    files: Annotated[
        list[UploadFile], 
        File(
            title="PDF Files", 
            description="PDF files to add to the vector database"
        )], 
    ):
    tasks = [process_data(file) for file in files]
    await asyncio.gather(*tasks)
    

@app.post("/generate/",
          summary="Given a question, this endpoint allows to generate an answer grounded on the documents in the VectorDB",
          tags=[Tags.LLM]
          )
async def get_answer(new_user_input: str, chat_history: list[ChatMsg]):
    mapping = {"human": HumanMessage, "assistant": AIMessage}

    msgs: list[HumanMessage | AIMessage] = None
    if chat_history:
        msgs = [mapping[chat_msg.role](content=chat_msg.message) for chat_msg in chat_history]
    
    answer_stream, *_ = ragPipe.run(question=new_user_input, chat_history_api=msgs)

    async def send_chunk():
        async for chunk in answer_stream:
            yield chunk.content
                  
    return StreamingResponse(send_chunk(), media_type="text/event-stream-")

    

mount_chainlit(app=app, target="app/chainlit_app.py", path="/chainlit")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) # 8083