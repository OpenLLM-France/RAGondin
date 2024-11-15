from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    MessagesPlaceholder, 
    ChatPromptTemplate
)
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from typing import AsyncIterator


class LLM:
    def __init__(self, config, logger=None):
        self.logger = logger
        self.client: ChatOpenAI = ChatOpenAI(
            model=config.llm["name"],
            base_url=config.llm["base_url"],
            api_key=config.llm['api_key'],
            timeout=60,
            temperature=config.llm["temperature"],
            max_tokens=config.llm["max_tokens"], 
            streaming=True,
        )   
         
    def run(self, 
            question: str, context: str, 
            chat_history: list[AIMessage | HumanMessage], 
            sys_pmpt_tmpl: str
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
                ("system", sys_pmpt_tmpl),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        rag_chain = qa_prompt | self.client.with_retry(stop_after_attempt=3)

        input_ = {
            "input": question, 
            "context": context,
            "chat_history": chat_history, 
        }
        return rag_chain.astream(input_)