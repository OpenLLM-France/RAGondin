from pathlib import Path
from .prompt import load_sys_template
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    MessagesPlaceholder, 
    ChatPromptTemplate
)
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import pprint as pp


# Define the Evaluator class
dir_path = Path(__file__).parent
metrics = {
    "contextual_relevancy": dir_path / "prompts/eval/contextual_relevancy_template.txt",
    "hallucination": dir_path / "prompts/eval/hallucination_template.txt"
}

def evaluate(llm: ChatOpenAI, context, chat_history, question, answer):
    # TODO: https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation
    for metric_name, path in metrics.items():
        metric_sys_msg = load_sys_template(path)
        metric_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", metric_sys_msg),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        history_aware_retriever = (
                metric_prompt
                | llm
                | JsonOutputParser()
            )
        
        input_ = {
            "input": question, "chat_history": list(chat_history), 
            "context": context, "generated_answer": answer
        }

        if metric_name == "hallucination":
            input_.update({"generated_answer": answer})

        eval_res = history_aware_retriever.invoke(input_)
        pp.pprint(f"{metric_name.title()}: {eval_res}")
        





        
    # def generate_questions(self, nb_questions: int, docs):
    #     """
    #     Generates a specified number of questions based on the documents.

    #     Parameters:
    #         nb_questions (int): The number of questions to be generated.
    #     """
    #     self.docs = docs
    #     self.questions = []

    #     random.seed(2000)
    #     sample_docs = random.sample(self.docs.get_chunks(), nb_questions)

    #     for sampled_context in tqdm(sample_docs, desc="Generating questions..."):
    #         output_QA_couple = self.llm.async_run(QA_generation_prompt.format(context=sampled_context.page_content)) # generate (question, answer) pairs
    #         try:
    #             # question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
    #             # answer = output_QA_couple.split("Answer: ")[-1]
    #             question, answer = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")
    #             assert len(answer) < 200, "Answer is too long"
    #             self.questions.append(
    #                 {
    #                     "context": sampled_context.page_content,
    #                     "question": question,
    #                     "answer": answer,
    #                     "source_doc": sampled_context.metadata["source"],
    #                 }
    #             )
    #         except:
    #             continue

    # def critique_questions(self):
    #     """
    #     Critiques the generated questions based on groundedness, relevance, and standalone criteria.
    #     """
    #     for question in tqdm(self.questions, desc="generate critiques..."):
    #         evaluations = {
    #             "groundedness": self.llm.async_run(
    #                 question_groundedness_critique_prompt.format(context=question["context"], question=question["question"])
    #             ),
    #             "relevance": self.llm.async_run(
    #                 question_relevance_critique_prompt.format(question=question["question"]),
    #             ),
    #             "standalone": self.llm.async_run(
    #                 question_standalone_critique_prompt.format(question=question["question"]),
    #             ),
    #         }
    #         for criterion, evaluation in evaluations.items():
    #             try:
    #                 eval, score = evaluation.strip().split("Evaluation: ")[-1].split("Total rating: ")
    #                 score = int(score.strip())
    #                 # print(f"===>{criterion}", score, eval, sep=" # ")
    #                 question.update(
    #                     {
    #                         f"{criterion}_score": score, 
    #                         f"{criterion}_eval": eval.strip()
    #                     }
    #                 )
    #             except Exception as e:
    #                 continue

            
    # def filter_questions(self, groundness_trsh: int = 1, relevance_trsh: int = 1, standalone_trsh: int = 1) -> None:
    #     """
    #     This method filters the generated questions based on the specified thresholds for groundedness, relevance, and standalone scores.
    #     It creates a DataFrame from the generated questions and filters it based on the thresholds. The filtered questions are then stored in the instance variable 'filtered_questions'.
    #     It also converts the filtered questions DataFrame into a Dataset and stores it in the instance variable 'eval_dataset'.

    #     Parameters:
    #         groundness_trsh (int): The minimum score a question must have in groundedness to be included in the filtered questions.
    #         relevance_trsh (int): The minimum score a question must have in relevance to be included in the filtered questions.
    #         standalone_trsh (int): The minimum score a question must have in standalone to be included in the filtered questions.
    #     """
    #     # print([len(q) for q in self.questions])

    #     # assert 1 == 2
    #     values = {"groundedness_score":1, "relevance_score":1, "standalone_score":1}
    #     generated_questions: pd.DataFrame = pd.DataFrame.from_dict(self.questions)
    #     generated_questions.fillna(value=values, inplace=True)

    #     # print(generated_questions)        
        
    #     c = (
    #         (generated_questions["groundedness_score"] >= groundness_trsh) & 
    #         (generated_questions["relevance_score"] >= relevance_trsh) & 
    #         (generated_questions["standalone_score"] >= standalone_trsh)
    #     )
        
    #     generated_questions = generated_questions.loc[c]
    #     self.filtered_questions: pd.DataFrame = generated_questions
    #     self.eval_dataset: datasets.Dataset = datasets.Dataset.from_pandas(generated_questions, split="train",
    #                                                                        preserve_index=False)

    # def eval_question_answer(self, question: str, answer: str, true_answer: str) -> tuple[str, int]:
    #     """
    #     This method evaluates the quality of a given answer to a question, compared to a reference answer.
    #     It uses the LLM model to generate an evaluation result, which is then parsed to extract the feedback and score.

    #     Parameters:
    #         question (str): The question for which the answer is provided.
    #         answer (str): The provided answer to be evaluated.
    #         true_answer (str): The reference answer to compare the provided answer against.

    #     Returns:
    #         tuple: A tuple containing the feedback as a string and the score as an integer.
    #     """
    #     eval_prompt = EVALUATION_PROMPT.format(
    #         instruction=question,
    #         response=answer,
    #         reference_answer=true_answer,
    #     )
    #     eval_result = self.llm.async_run(eval_prompt).split("###Feedback")[-1]
    
    #     try:
    #         feedback, score =  eval_result.split("[RESULT]: ") # [item.strip() for item in eval_result.split("[RESULT]")]
    #         return {"feedback":feedback.strip(), "score": score.strip()}
    #     except Exception as e:
    #         return {"feedback":"", "score": ""}

    # def evaluate(self, rag):
    #     """
    #     This method evaluates the RAG model based on the filtered questions.
    #     It generates answers for the filtered questions using the RAG model, then evaluates the quality of these answers compared to the reference answers.
    #     The evaluation is based on the feedback and score obtained from the `eval_question_answer` method.
    #     The results are saved in a dataset and returned.

    #     Parameters:
    #         rag (RAG): The RAG model to be evaluated.

    #     Returns:
    #         datasets.Dataset: The dataset containing the questions, context, reference answers, generated answers, feedback, and scores.
    #     """
    #     eval_dataset = self.eval_dataset

    #     eval_dataset = eval_dataset.map(
    #         lambda example: {
    #             "question": example["question"],
    #             "context": example["context"],
    #             "answer": example["answer"],
    #             "source_doc": example["source_doc"],
    #             "generated_output": rag.run(example["question"]),
    #         }
    #     )
        
    #     eval_dataset = eval_dataset.map(
    #         lambda example: {
    #             "question": example["question"],
    #             "context": example["context"],
    #             "answer": example["answer"],
    #             "source_doc": example["source_doc"],
    #             "generated_output": example["generated_output"],
    #             **self.eval_question_answer(example["question"], example["generated_output"], example["answer"])
    #         }
    #     )
    #     return eval_dataset
