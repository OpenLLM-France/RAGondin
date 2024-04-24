from typing import Union
from transformers.pipelines.text_generation import TextGenerationPipeline
from huggingface_hub import InferenceClient
import json

class LLM:
    """
    A class to represent a Language Model (LLM).
    It abstracts away the model type and provides a unified interface for the user.
    ...

    Attributes
    ----------
    model : Union[TextGenerationPipeline, InferenceClient]
        a transformers pipeline or a Hugging Face InferenceClient

    Methods
    -------
    generate_output(prompt):
        Generates output from the given prompt using the model.
    """

    def __init__(self, model: Union[TextGenerationPipeline, InferenceClient]):
        """
        Constructs all the necessary attributes for the LLM object.

        Parameters
        ----------
            model : Union[TextGenerationPipeline, InferenceClient]
                a transformers pipeline or a Hugging Face InferenceClient
        """
        if isinstance(model, TextGenerationPipeline):
            self.model = model
        elif isinstance(model, InferenceClient):
            self.model = model
        else:
            raise ValueError("Model should be either a transformers pipeline or a Hugging Face InferenceClient")

    def run(self, prompt: str) -> str:
        """
        Generates output from the given prompt using the model.

        Parameters
        ----------
            prompt : str
                the input prompt

        Returns
        -------
            str
                the generated output from the model
        """
        if isinstance(self.model, TextGenerationPipeline):
            return self.model(prompt)[0]["generated_text"]
        elif isinstance(self.model, InferenceClient):
            return call_llm(self.model, prompt)



def call_llm(inference_client: InferenceClient, prompt: str):
    """
    Calls the LLM with the given inference client and prompt.

    Parameters
    ----------
        inference_client : InferenceClient
            the Hugging Face InferenceClient
        prompt : str
            the input prompt

    Returns
    -------
        str
            the generated output from the LLM
    """
    response = inference_client.post(
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1000},
            "task": "text-generation",
        },
    )
    return json.loads(response.decode())[0]["generated_text"]