from langchain.prompts import PromptTemplate
from .llm import LLM
from pathlib import Path
from pathlib import Path
from langchain_core.documents.base import Document


dir_path = Path(__file__).parent

class BasicPrompt:
    """Class for generating prompts to provide context for LLMs."""
    def __init__(
            self, 
            prompt_file: str = dir_path / "prompts/basic_sys_prompt_template.txt"
        ) -> None:
        
        self.sys_template: str = get_sys_template(prompt_file)



        
class MultiQueryPrompt(BasicPrompt):
    def __init__(
            self, 
            prompt_file: str = dir_path / "prompts/multi_query_prompt_template.txt",
        ) -> None:
        super().__init__(prompt_file)
    
    def get_multi_query_prompt(self, question: str, k_queries: int) -> dict:
        sys_template, user_template = self.sys_template
        sys_prompt = sys_template.format(k=str(k_queries))
        user_prompt = user_template.format(question=question)
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
    


PROMPT_TEMPLATES = {
    "basic": BasicPrompt,
    "multiQuery": MultiQueryPrompt
}   

def get_sys_template(file_path: Path) -> tuple[str, str]:
    with open(file_path, mode="r") as f:
        sys_msg = f.read()
        return sys_msg


def format_context(docs: list[Document]) -> str:
    """Build context string from list of documents."""
    # TODO: Add links to used document and specify to the model (in the prompt) to use them for referencing.
    context = "Extracted documents:\n"
    for i, doc in enumerate(docs, start=1):
        context += f"""
        Document: {doc.page_content}
        source{i}: {doc.metadata["source"]}#page={doc.metadata["page"]+1}
        =======\n
        """
    return context


def generate_multi_query(llm: LLM, msg_prompts: dict) -> list[str]:
    questions = llm.run(msg_prompts)
    questions = questions.split("####Generated Questions: ")[-1]
    questions_l = questions.split("[SEP]")
    return questions_l