from resume_parser import ResumeParser
from question_generator import QuestionGenerator
from interviewer import Interviewer
from evaluator import Evaluator


class AIInterviewer:
    def __init__(self):
        self.resume_parser = ResumeParser()
        self.question_generator = QuestionGenerator()
        self.interviewer = Interviewer()
        self.evaluator = Evaluator()

    def conduct_interview(self, resume_path):
        # 解析简历
        projects = self.resume_parser.parse(resume_path)

        # 生成问题
        questions = self.question_generator.generate_questions(projects)

        # 进行面试
        responses = self.interviewer.interview(questions)

        # 评估表现
        evaluation = self.evaluator.evaluate(responses)

        return evaluation


# 使用示例
if __name__ == "__main__":
    ai_interviewer = AIInterviewer()
    result = ai_interviewer.conduct_interview("path/to/resume.pdf")
    print(result)
