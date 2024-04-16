import configparser
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from vector_store import Qdrant_Connector


#from langchain.document_loaders import PyPDFLoader
from embeddings import Embeddings


# Read config file
config = configparser.ConfigParser()
config.read("config.ini")

host = config.get("VECTOR_DB", "host")
port = config.get("VECTOR_DB", "port")
collection = config.get("VECTOR_DB", "collection")
print(collection)
model_type = config.get("EMBEDDINGS", "model_type")
model_name = config.get("EMBEDDINGS", "model_name")
model_kwargs = dict(config.items("EMBEDDINGS.MODEL_KWARGS"))
encode_kwargs = dict(config.items("EMBEDDINGS.ENCODE_KWARGS"))



#TODO: read data directory from confing file
#TODO: create splitter class 
loader = PyPDFLoader("../data/gpt4.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# Load the embedding model 
embeddings = Embeddings(model_type, model_name, model_kwargs, encode_kwargs).get_embeddings()

url = f"http://{host}:{port}"
connector = Qdrant_Connector(host, port, collection, embeddings)
connector.build_index(chunks)


#TODO: use loguru library for logging
print("Collection in QDrant DB successfully created!")

