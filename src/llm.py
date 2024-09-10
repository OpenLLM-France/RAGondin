from typing import Union
from transformers.pipelines.text_generation import TextGenerationPipeline
from huggingface_hub import InferenceClient
import json
from openai import OpenAI, AsyncOpenAI, AsyncStream

class LLM:
    def __init__(
            self, 
            client: OpenAI | AsyncOpenAI,
            model_name: str = 'meta-llama-31-8b-it',
            max_tokens: int = 1000
        ):
        """This class implements logics to call an OpenAI client with respect to different contexts

        Args:
            client (OpenAI | AsyncOpenAI): Your OpenAI client initialized with your api_key, base_url, etc.
            model_name (str, optional): The name of your remote model. Defaults to 'meta-llama-31-8b-it'.
            max_tokens (int, optional): Max tokens to generate. Defaults to 1000.

        Raises:
            ValueError: If the `model_type` is not OpenAI.
        """
        if isinstance(client, (OpenAI, AsyncOpenAI)):
            self.client = client
            self.model_name = model_name
            self.max_tokens = max_tokens
        else:
            raise ValueError(f"Model should be of type {OpenAI}")
    
    async def async_run(self, prompt_dict: dict) -> AsyncStream:
        # TODO: Transform it into a classmethod maybe
        """Method for chat complettion in `streaming` mode. This method is then Async.
        Use it when the client is of type `AsyncOpenAI`.

        Args:
            prompt_dict (dict): Prompt dict contains `system` and `user` content 
            to fill up the `messages` argument of the `OpenAI.chat.completions.create` method.

        Returns:
            str: Return an Async stream.
        """
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
        """This method Chat completion. To use when when the client is of type `OpenAI`.

        Args:
            prompt_dict (dict): Prompt dict contains `system` and `user` content 
            to fill up the `messages` argument of the `OpenAI.chat.completions.create` method.

        Returns:
            str: Returns the answer of the LLM in string format
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompt_dict["system"]},
                {"role": "user", "content": prompt_dict["user"]}
            ],
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
    
