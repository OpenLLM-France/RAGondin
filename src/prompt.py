from langchain.prompts import PromptTemplate

BASIC_PROMPT_TEMPLATE = """
<|system|>
Answer the question in french only using the following french context to help:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>

 """




class Prompt:
    def __init__(self, prompt_template: PromptTemplate = None:
        if prompt_template:
            self.prompt_template = prompt_template
        else :
            self.prompt_template = PromptTemplate(input_variables=["context", "question"],template=BASIC_PROMPT_TEMPLATE)

    def get_prompt(self, docs: list[str]=[""], question="") -> str:
        context = "\nExtracted documents:\n"
        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(docs)])
        return self.prompt_template.format(question=question, context=context)

        
        
    