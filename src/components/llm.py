from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    MessagesPlaceholder, 
    ChatPromptTemplate
)
from langchain_core.messages import AIMessage, HumanMessage


class LLM2:
    def __init__(
            self, 
            model_name: str = 'meta-llama-31-8b-it',
            base_url: str = None, 
            api_key: str = None,
            timeout: int = 60,
            max_tokens: int = 1000,
        ):

        self.client: ChatOpenAI = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_tokens=max_tokens, 
            streaming=True
        )    

    def run(self, 
            question: str, context: str, 
            chat_history: list[AIMessage | HumanMessage], 
            sys_prompt_template: str
        ):
        """This method runs the algorithm 

        Args:
            question (str): The input from the user; not necessarily a question.
            context (str): It's the retrieved documents
            chat_history (list[AIMessage  |  HumanMessage]): The Chat history.
            sys_prompt_template (str): _description_

        Returns:
            _type_: _description_
        """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sys_prompt_template),
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
        
        

def template_from_sys_template(sys_message) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
            [
                ("system", sys_message),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )