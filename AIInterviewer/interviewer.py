from typing import Literal, Optional
from .evaluator import Evaluator


class Interviewer:
    def __init__(self, questions_from_cv):
        self.questions_from_cv = questions_from_cv
        self.current_question_index = 0
        self.evaluator = Evaluator()
        self.tracker = []

    def fetch_question(
        self,
        mode: Literal["probe", "new"],
        evaluation: Optional[dict] = None,
    ):
        # probe: 追问
        # new: 新问题
        if mode == "new":
            question = self.questions_from_cv[self.current_question_index]
            self.current_question_index += 1
        elif mode == "probe":
            if evaluation is None:
                raise ValueError("Evaluation is required for probe mode")
            pass
        return question

    def process_response(self, response: str):
        evaluation = self.evaluator.evaluate(response)
        if evaluation["passed"]:
            question = self.fetch_question("new")
        else:
            question = self.fetch_question("probe", evaluation)
