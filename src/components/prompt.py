from langchain.prompts import PromptTemplate
from .llm import LLM
from pathlib import Path
from pathlib import Path


dir_path = Path(__file__).parent

class BasicPrompt:
    """Class for generating prompts to provide context for LLMs."""
    def __init__(
            self, 
            prompt_file: str = dir_path / "prompts/basic_prompt_template.txt"
        ) -> None:
        
        self.msg_templates = get_msgs(prompt_file)

    def get_prompt(self, question: str, docs: list[str] = None) -> tuple[dict, str]:
        context = _build_context(docs)
        sys_template, user_template = self.msg_templates
        sys_prompt = sys_template.format(context=context)
        user_prompt = user_template.format(question=question)
        return {"system":sys_prompt, "user":user_prompt}, context
        
class MultiQueryPrompt(BasicPrompt):
    def __init__(
            self, 
            prompt_file: str = dir_path / "prompts/multi_query_prompt_template.txt",
        ) -> None:
        super().__init__(prompt_file)
    
    def get_multi_query_prompt(self, question: str, k_queries: int) -> dict:
        sys_template, user_template = self.msg_templates
        sys_prompt = sys_template.format(k=str(k_queries))
        user_prompt = user_template.format(question=question)
        return {"system":sys_prompt, "user":user_prompt}
    


PROMPT_TEMPLATES = {
    "basic": BasicPrompt,
    "multiQuery": MultiQueryPrompt
}   

def get_msgs(file_path: Path) -> tuple[str, str]:
    with open(file_path, mode="r") as f:
        txt = f.read()
        sys_msg, user_msg = txt.split("&&&\n")
        return sys_msg, user_msg

def _build_context(docs: list[str]) -> str:
    """Build context string from list of documents."""
    #TODO: Add links to used document and specify to the model (in the prompt) to use them for referencing.
    context = "Extracted documents:\n"
    for i, doc in enumerate(docs):
        context += f"Document {i}:::\n{doc}\n"
    return context

def generate_multi_query(llm: LLM, msg_prompts: dict) -> list[str]:
    questions = llm.run(msg_prompts)
    questions = questions.split("####Generated Questions: ")[-1]
    questions_l = questions.split("[SEP]")
    return questions_l


# class Prompt:
#     """Class for generating prompts to provide context for LLMs."""

#     def __init__(self, type_template: str = 'basic') -> None:
#         """
#         Initialize Prompt.

#         Args:
#             type_template (str): Type of template to use. It can be either "basic" or "multiQuery".
#         """

#         dir_path = Path(__file__).parent
#         # TODO: Avoid hard coding paths.
#         if type_template == TemplateType.BASIC:
#             msg_templates = get_msgs(dir_path / "prompts/basic_prompt_template.txt")

#         elif type_template == TemplateType.MULTI_QUERY:
#             msg_templates = get_msgs(dir_path / "prompts/multi_query_prompt_template.txt")
#         else:
#             raise ValueError(f"This `template_type` isn't supported")
        
#         self.typed_template = type_template
#         self.msg_templates = msg_templates


#     def get_prompt(self, question: str, docs: list[str] = None, k_multi_queries: int = 3) -> dict:
#         """
#         Generate prompt from documents and question.

#         Args:
#             docs (list[str]): List of document strings.
#             question (str): Question string in the case of *type_template* is "basic".
#             k_multi_queries (int): Number of multi queries to generate in the case of *type_template* is "multiQuery".
#         Returns:
#             str: Final generated prompt.
#         """

#         if self.typed_template == TemplateType.BASIC: # QA mode, we return the prompt_dict for the llm and the built context
#             context = _build_context(docs)
#             sys_template, user_template = self.msg_templates
#             sys_prompt = sys_template.format(context=context)
#             user_prompt = user_template.format(question=question)
#             return {"system":sys_prompt, "user":user_prompt}, context
        

#         if self.typed_template == TemplateType.MULTI_QUERY:
#             sys_template, user_template = self.msg_templates
#             sys_prompt = sys_template.format(k=str(k_multi_queries))
#             user_prompt = user_template.format(question=question)
#             return {"system":sys_prompt, "user":user_prompt}
            