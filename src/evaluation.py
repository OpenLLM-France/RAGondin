import random
from typing import Union, List

from src.chunker import Docs
from src.llm import LLM

import pandas as pd
import datasets

from src.pipeline import RAG

EVALUATION_PROMPT = """
###Context:
You are a fair evaluator language model for french documentation.
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""

question_groundedness_critique_prompt = """
You will be given a context and a question both in french.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """

question_relevance_critique_prompt = """
You will be given a question both in french.
Your task is to provide a 'total rating' representing how useful this question can be to machine learning developers building NLP applications with the Hugging Face ecosystem.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """

question_standalone_critique_prompt = """
You will be given a question both in french
Your task is to provide a 'total rating' representing how context-independant this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independant from the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """


QA_generation_prompt = """
Your task is to write a factoid question in french and an answer in french given a context in french.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::"""

class Evaluator:
    def __init__(self,save_path:str, llm: LLM, docs: Docs):
        """
            The constructor for the Evaluator class.

            Parameters:
                save_path (str): The path where the evaluation data_set.csv will be saved.
                llm (LLM): An instance of the LLM class used for generating outputs.
                docs (Docs): An instance of the Docs class containing the documents to be evaluated.
        """
        self.save_path = save_path
        self.llm = llm
        self.docs = docs
        self.questions: Union[List[str], None] = None
        self.filtered_questions: Union[pd.DataFrame, None] = None
        self.eval_dataset : Union[datasets.Dataset,None] = None

    def generate_questions(self, nb_questions:int):
        """
        Generates a specified number of questions based on the documents.

        Parameters:
            nb_questions (int): The number of questions to be generated.
        """
        self.questions = []
        sample_docs = random.sample(self.docs.get_chunks(), nb_questions)

        for sampled_context in sample_docs:
            output_QA_couple = self.llm.run(QA_generation_prompt.format(context=sampled_context.page_content))
            try:
                question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
                answer = output_QA_couple.split("Answer: ")[-1]
                assert len(answer) < 300, "Answer is too long"
                self.questions.append(
                    {
                        "context": sampled_context.page_content,
                        "question": question,
                        "answer": answer,
                        "source_doc": sampled_context.metadata["source"],
                    }
                )
            except:
                continue
    def critique_questions(self):
        """
        Critiques the generated questions based on groundedness, relevance, and standalone criteria.
        """
        for question in self.questions:
            evaluations = {
                "groundedness": self.llm.run(question_groundedness_critique_prompt.format(context=question["context"], question=question["question"]),
                                             ),
                "relevance": self.llm.run(
                    question_relevance_critique_prompt.format(question=question["question"]),
                ),
                "standalone": self.llm.run(
                    question_standalone_critique_prompt.format(question=question["question"]),
                ),
            }
            try:
                for criterion, evaluation in evaluations.items():
                    score, eval = (
                        int(evaluation.split("Total rating: ")[-1].strip()),
                        evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
                    )
                    question.update(
                        {
                            f"{criterion}_score": score,
                            f"{criterion}_eval": eval,
                        }
                    )
            except Exception as e:
                continue

    def filter_questions(self, groundness_trsh:int = 1, relevance_trsh:int = 1, standalone_trsh:int =1) -> None:
        """
        Filters the critiqued questions based on the specified thresholds for groundedness, relevance, and standalone scores.

        Parameters:
            groundness_trsh (int): The threshold for the groundedness score.
            relevance_trsh (int): The threshold for the relevance score.
            standalone_trsh (int): The threshold for the standalone score.
        """
        generated_questions: pd.DataFrame = pd.DataFrame.from_dict(self.questions)

        generated_questions = generated_questions.loc[
            (generated_questions["groundedness_score"] >= groundness_trsh)
            & (generated_questions["relevance_score"] >= relevance_trsh)
            & (generated_questions["standalone_score"] >= standalone_trsh)
            ]
        self.filtered_questions : pd.DataFrame = generated_questions
        self.eval_dataset : datasets.Dataset = datasets.Dataset.from_pandas(generated_questions, split="train", preserve_index=False)


    def eval_question_answer(self, question:str, answer:str, true_answer:str) -> tuple[str, int]:
        """
        Evaluates the quality of a given answer to a question, compared to a reference answer.

        Parameters:
            question (str): The question for which the answer is provided.
            answer (str): The provided answer to be evaluated.
            true_answer (str): The reference answer to compare the provided answer against.

        Returns:
            tuple: A tuple containing the feedback as a string and the score as an integer.
        """
        eval_prompt = EVALUATION_PROMPT.format(
            instruction=question,
            response=answer,
            reference_answer=true_answer,
        )
        eval_result = self.llm.run(eval_prompt).split("###Feedback")[-1]

        feedback, score = [item.strip() for item in eval_result.split("[RESULT]")]
        return feedback, score

    def evaluate(self, rag : RAG):
        """
        Evaluates the RAG model based on the filtered questions.

        This method generates answers for the filtered questions using the RAG model,
        then evaluates the quality of these answers compared to the reference answers.
        The evaluation is based on the feedback and score obtained from the `eval_question_answer` method.
        The results are saved in a dataset and stored in the specified save path.

        Parameters:
            rag (RAG): The RAG model to be evaluated.

        Returns:
            datasets.Dataset: The dataset containing the questions, context, reference answers, generated answers, feedback, and scores.
        """
        eval_dataset = self.eval_dataset
        eval_dataset = eval_dataset.map(
            lambda example: {
                "question": example["question"],
                "context": example["context"],
                "answer": example["answer"],
                "source_doc": example["source_doc"],
                "generated_output": rag.run(example["question"]),
            }
        )
        eval_dataset = eval_dataset.map(
            lambda example: {
                "question": example["question"],
                "context": example["context"],
                "answer": example["answer"],
                "source_doc": example["source_doc"],
                "generated_output": example["generated_output"],
                "feedback": self.eval_question_answer(
                    example["question"], example["generated_output"], example["answer"]
                )[0],
                "score": self.eval_question_answer(
                    example["question"], example["generated_output"], example["answer"]
                )[1],
            }
        )
        eval_dataset.save_to_disk(self.save_path)
        return eval_dataset
