from langchain_community.embeddings import HuggingFaceBgeEmbeddings

class Embeddings:
    def __init__(self, model_type, model_name, model_kwargs, encode_kwargs):
        if model_type == "huggingface_bge":
            self.embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        else:
            print(f"{model_type} is not a correct type. Please, provide correct model_type.")

    def get_embeddings(self):
        return self.embeddings

