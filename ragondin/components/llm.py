from typing import AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


class LLM:
    def __init__(self, llm_config, logger=None):
        self.logger = logger
        print()
        self.client: ChatOpenAI = ChatOpenAI(**llm_config)

    def run(
        self,
        question: str,
        context: str,
        chat_history: list[AIMessage | HumanMessage],
        sys_pmpt_tmpl: str,
    ) -> AsyncIterator[BaseMessage]:
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
                ("human", "{input}"),
            ]
        )
        rag_chain = qa_prompt | self.client.with_retry(stop_after_attempt=3)

        input_ = {
            "input": question,
            "context": context,
            "chat_history": chat_history,
        }
        return rag_chain.astream(input_)
