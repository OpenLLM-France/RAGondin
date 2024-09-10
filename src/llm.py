from typing import Union
from transformers.pipelines.text_generation import TextGenerationPipeline
from huggingface_hub import InferenceClient
import json
from openai import OpenAI, AsyncOpenAI

class LLM:
    def __init__(
            self, 
            client: OpenAI | AsyncOpenAI,
            model_name: str = 'meta-llama-31-8b-it',
            max_tokens: int = 1000
        ):
        if isinstance(client, (OpenAI, AsyncOpenAI)):
            self.client = client
            self.model_name = model_name
            self.max_tokens = max_tokens
        else:
            raise ValueError(f"Model should be of type {OpenAI}")

    async def async_run(self, prompt_dict: dict) -> str:
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt_dict["system"]},
                {"role": "user", "content": prompt_dict["user"]}
            ],
            stream=True,
            max_tokens=self.max_tokens
        )
        return stream
        
    def run(self, prompt_dict: dict):
        response = self.client.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt_dict["system"]},
                {"role": "user", "content": prompt_dict["user"]}
            ],
            max_tokens=self.max_tokens,
            stream=False
        )
        return response.choices[0].text