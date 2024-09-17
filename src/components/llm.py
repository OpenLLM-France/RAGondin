from typing import Literal, Union
from transformers.pipelines.text_generation import TextGenerationPipeline
from openai import OpenAI, AsyncOpenAI, AsyncStream
from langchain_openai import ChatOpenAI
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate


class LLM2:
    def __init__(
            self, 
            model_name: str = 'meta-llama-31-8b-it',
            base_url: str = "https://chat.ai.linagora.exaion.com/v1/", 
            api_key: str = '',
            timeout: int = 60,
            max_tokens: int = 1000,
            streaming=True
        ):

        self.client: ChatOpenAI = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_tokens=max_tokens, streaming=streaming
        )    

    def run(self, 
            question: str, 
            context: str, 
            chat_history: list, 
            sys_msg: str
        ):

        qa_prompt = template_from_sys_template(sys_msg)
        rag_chain = (
            qa_prompt
            | self.client
        )
        input_ = {
            "input": question, 
            "context": context,
            "chat_history": chat_history, 
        }
        answer = rag_chain.astream(input_)
        return answer
        


def template_from_sys_template(sys_message) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
            [
                ("system", sys_message),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

            

    





class LLM:
    def __init__(
            self, 
            client: OpenAI | AsyncOpenAI,
            model_name: str = 'meta-llama-31-8b-it',
            max_tokens: int = 1000,
            chat_mode: Literal["ChatBotRag", "SimpleLLM"] = "SimpleLLM"
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
            self.chat_mode = chat_mode
            splf._messages: list = None
        else:
            raise ValueError(f"Model should be of type {OpenAI}")
    
    def update_history(self, prompt_msg: list[dict]):
        if self._messages is None:
            self._messages = []
            self._messages.extend(prompt_msg)
        else:       
            user_msg = prompt_msg[-1]
            self._messages.append(user_msg)

    
    async def async_run(self, prompt_msg: list[dict]):
        """Method for chat complettion in `streaming` mode. This method is then Async.
        Use it when the client is of type `AsyncOpenAI`.

        Args:
            prompt_dict (dict): Prompt dict contains `system` and `user` content 
            to fill up the `messages` argument of the `OpenAI.chat.completions.create` method.

        Returns:
            str: Return an Async stream.
        """
        self.update_history(prompt_msg) # add the user and potentially system + user message

        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=self._messages,
            stream=True,
            max_tokens=self.max_tokens
        )

        full_answer = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                d = chunk.choices[0].delta.content
                full_answer += d
                yield d
        
        self._messages.append({"role":"assistant", "content": full_answer}) # add the answer

    
    def run(self, prompt_msg: list[dict]) -> str:
        """This method Chat completion. To use when when the client is of type `OpenAI`.

        Args:
            prompt_dict (dict): Prompt dict contains `system` and `user` content 
            to fill up the `messages` argument of the `OpenAI.chat.completions.create` method.

        Returns:
            str: Returns the answer of the LLM in string format
        """
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt_msg,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
    
