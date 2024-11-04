from fastapi import FastAPI
from chainlit.utils import mount_chainlit
from starlette.requests import Request
from starlette.responses import RedirectResponse

app = FastAPI()

mount_chainlit(app=app, target="./chainlit_app.py", path="/chainlit")
# uvicorn chainlit_app2:app --host 0.0.0.0 --port 8000 --reload