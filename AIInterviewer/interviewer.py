import json
from typing import Literal, Optional, List, Iterator
from .evaluator import Evaluator, EvaluationResult
from .prompts import ERROR_HINT_PROMPT, INCOMPLETE_PROBE_PROMPT, INTERVIEW_SYSTEM_PROMPT
from .llm_client import get_openai_client
from loguru import logger


def question_generator(
    questions_from_cv: List[dict],
    max_project_questions: int,
    max_keypoints: int,
    max_questions_per_keypoint: int,
    load_path: Optional[str] = None,
) -> Iterator[dict]:
    if load_path:
        with open(load_path, "r") as f:
            questions_from_cv = json.load(f)

    # 生成reform之后的questions的生成器，每个项目最多max_project_questions个项目问题，以及max_keypoint_questions个关键点问题
    for questions_per_project in questions_from_cv:
        for question in questions_per_project["project_questions"][
            :max_project_questions
        ]:
            yield dict(
                project=questions_per_project["project"],
                keypoint=None,
                references=None,
                question=question,
            )
        for keypoint in questions_per_project["keypoint_questions"][:max_keypoints]:
            for question in keypoint["questions"][:max_questions_per_keypoint]:
                yield dict(
                    project=questions_per_project["project"],
                    keypoint=keypoint["keypoint"],
                    references=keypoint["references"],
                    question=question,
                )


class Interviewer:
    def __init__(
        self,
        questions_from_cv: List[dict],
        api_type: Literal["azure", "openai", "qwen", "ollama"],
    ):
        self.questions_from_cv = question_generator(
            questions_from_cv=questions_from_cv,
            max_project_questions=2,
            max_keypoints=2,
            max_questions_per_keypoint=1,
        )
        self.evaluator = Evaluator(api_type=api_type)
        self.model, self.client = get_openai_client(api_type=api_type)

    def generate_error_hint(
        self, question: dict, response: str, comment: Optional[str]
    ) -> str:
        """生成错误提示"""
        if comment is None:
            return ""

        prompt = ERROR_HINT_PROMPT.format(
            project=question["project"],
            keypoint=question["keypoint"],
            references=question["references"],
            question=question["question"],
            response=response,
            comment=comment,
        )

        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": INTERVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = result.choices[0].message.content
        return content if content is not None else ""

    def generate_probe_question(
        self, question: dict, response: str, comment: Optional[str]
    ) -> str:
        """生成追问问题"""
        if comment is None:
            return ""

        prompt = INCOMPLETE_PROBE_PROMPT.format(
            project=question["project"],
            keypoint=question["keypoint"],
            references=question["references"],
            question=question["question"],
            response=response,
            comment=comment,
        )

        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": INTERVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        content = result.choices[0].message.content
        return content if content is not None else ""

    def fetch_question(
        self,
        mode: Literal["next", "stay"],
        evaluation: Optional[EvaluationResult] = None,
        current_question: Optional[dict] = None,
        response: Optional[str] = None,
    ) -> dict:
        if mode == "next":
            try:
                return next(self.questions_from_cv)
            except StopIteration:
                return dict()

        elif mode == "stay":
            if evaluation is None or current_question is None or response is None:
                raise ValueError(
                    "Evaluation, current question and response are required for stay mode"
                )

            if evaluation.eval_status == 1:
                # 回答错误，生成提示
                hint = self.generate_error_hint(
                    current_question, response, evaluation.comment
                )
                if hint:
                    new_question_dict = current_question.copy()
                    new_question_dict["question"] = hint
                    return new_question_dict
                else:
                    return self.fetch_question("next")

            elif evaluation.eval_status == 2:
                # 回答不完整，生成追问
                probe = self.generate_probe_question(
                    current_question, response, evaluation.comment
                )
                if probe:
                    new_question_dict = current_question.copy()
                    new_question_dict["question"] = probe
                    return new_question_dict
                else:
                    return self.fetch_question("next")

            # 如果没有命中上面的条件，返回下一个问题
            try:
                return next(self.questions_from_cv)
            except StopIteration:
                return dict()

    def process_response(self, question: dict, response: str):
        evaluation = self.evaluator.evaluate(question, response)
        logger.info(f"{evaluation=}")
        if evaluation.passed:
            next_question = self.fetch_question("next")
        else:
            next_question = self.fetch_question(
                "stay",
                current_question=question,
                response=response,
                evaluation=evaluation,
            )
        return next_question
