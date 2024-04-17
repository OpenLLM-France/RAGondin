from transformers import pipeline
from huggingface_hub import InferenceClient


class LLM:
    def __init__(self, model):
        if isinstance(model, pipeline):
            self.model = model
        elif isinstance(model, InferenceClient):
            self.model = model
        else:
            raise ValueError("Model should be either a transformers pipeline or a Hugging Face InferenceClient")

    def generate_output(self, prompt):
        if isinstance(self.model, pipeline):
            return self.model(prompt)[0]["generated_text"]
        elif isinstance(self.model, InferenceClient):
            return self.model = model