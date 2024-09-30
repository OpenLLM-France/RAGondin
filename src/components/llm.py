from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    MessagesPlaceholder, 
    ChatPromptTemplate
)
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from typing import AsyncIterator

class LLM:
    def __init__(
            self, 
            model_name: str = 'meta-llama-31-8b-it',
            base_url: str = None, 
            api_key: str = None,
            timeout: int = 60,
            max_tokens: int = 1000,
        ):
        # https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683

        self.client: ChatOpenAI = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_tokens=max_tokens, 
            streaming=True,
            # temperature=0.3,
        )    

    def run(self, 
            question: str, context: str, 
            chat_history: list[AIMessage | HumanMessage], 
            sys_prompt_tmpl: str
        )-> AsyncIterator[BaseMessage]:

        """This method runs the LLM given the user's input (`question`), `chat_history`, and the system prompt template (`sys_prompt_tmpl`) 

        Args:
            question (str): The input from the user; not necessarily a question.
            context (str): It's the retrieved documents (formatted into a string)
            chat_history (list[AIMessage  |  HumanMessage]): The Chat history.
            sys_prompt_tmpl (str): The system prompt
        Returns:
            AsyncIterator[BaseMessage]
        """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sys_prompt_tmpl),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        rag_chain = (
            qa_prompt
            | self.client
        )

        input_ = {
            "input": question, "context": context,
            "chat_history": chat_history, 
        }
        answer = rag_chain.astream(input_)    
        return answer