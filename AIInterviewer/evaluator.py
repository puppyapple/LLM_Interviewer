from typing import Optional, Literal, Dict
from pydantic import BaseModel
import instructor
from loguru import logger
from .llm_client import get_openai_client
from .prompts import EVALUATION_SYSTEM_PROMPT, EVALUATION_PROMPT


class EvaluationResult(BaseModel):
    eval_status: Literal[0, 1, 2, 3]  # 0: 正确, 1: 错误, 2: 待细化, 3: 不知道
    comment: Optional[str] = None

    @property
    def passed(self) -> bool:
        """用于兼容interviewer.py中的判断"""
        return self.eval_status == 0 or self.eval_status == 3

    def to_dict(self) -> Dict:
        """用于兼容interviewer.py中的参数传递"""
        return {"eval_status": self.eval_status, "comment": self.comment}


class Evaluator:
    def __init__(self, api_type: str = "ollama"):
        model, openai_client = get_openai_client(api_type)
        self.model = model
        self.client = instructor.from_openai(openai_client, mode=instructor.Mode.JSON)

    def evaluate(self, question: dict, response: str) -> EvaluationResult:
        logger.info(f"{question=}")
        logger.info(f"{response=}")
        prompt = EVALUATION_PROMPT.format(
            project=question["project"],
            keypoint=question.get("keypoint", "无"),
            references=question.get("references", "无"),
            question=question["question"],
            response=response,
        )

        evaluation = self.client.chat.completions.create(
            model=self.model,
            response_model=EvaluationResult,
            messages=[
                {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        return evaluation
