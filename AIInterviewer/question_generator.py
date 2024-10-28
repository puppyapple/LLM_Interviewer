from icecream import ic
from .knowledge_base import KnowledgeBase
from .prompts import (
    KEY_POINTS_EXTRACTION_PROMPT,
    KEYPOINT_QUESTION_GENERATION_PROMPT,
    INTERVIEW_SYSTEM_PROMPT,
    RESUME_ANALYSIS_SYSTEM_PROMPT,
    PROJECT_QUESTION_GENERATION_PROMPT,
)
from .llm_client import get_openai_client
from functools import lru_cache


class QuestionGenerator:
    def __init__(self, kb: KnowledgeBase, api_type="ollama"):
        self.knowledge_base = kb
        self.model, self.client = get_openai_client(api_type)  # 初始化OpenAI客户端

    def extract_key_points(self, project):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RESUME_ANALYSIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": KEY_POINTS_EXTRACTION_PROMPT.format(project=project),
                },
            ],
        )
        return response.choices[0].message.content.split("\n")

    def retrieve_knowledge(self, keypoints):
        knowledges = []
        for keypoint in keypoints:
            if not keypoint:
                knowledges.append(None)
                continue
            relevant_docs = self.knowledge_base.query(keypoint).source_nodes
            if not relevant_docs:
                knowledges.append(None)
                continue
            context = relevant_docs[0].node.metadata["window"]
            knowledges.append(context)
        return knowledges

    @lru_cache(maxsize=100)
    def generate_project_questions(self, project):
        # 生成项目粒度的面试问题
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": INTERVIEW_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": PROJECT_QUESTION_GENERATION_PROMPT.format(
                        project=project
                    ),
                },
            ],
        )
        res = [_ for _ in response.choices[0].message.content.split("\n") if _]
        return res

    @lru_cache(maxsize=100)
    def generate_keypoint_questions(self, project, keypoint, relevant_knowledge):
        # 生成技术关键点粒度的面试问题
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": INTERVIEW_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": KEYPOINT_QUESTION_GENERATION_PROMPT.format(
                        project=project,
                        keypoint=keypoint,
                        relevant_knowledge=relevant_knowledge,
                    ),
                },
            ],
        )
        res = [_ for _ in response.choices[0].message.content.split("\n") if _]
        return res

    def generate_questions(self, projects):
        question_list = []
        for project in projects:
            questions = dict()
            questions["project"] = project

            # 生成面试问题
            project_questions = self.generate_project_questions(project)
            questions["project_questions"] = project_questions

            # 提取项目的技术关键点
            key_points = self.extract_key_points(project)
            if not key_points:
                questions["keypoint_questions"] = None
                continue
            questions["keypoint_questions"] = []

            relevant_knowledges = self.retrieve_knowledge(key_points)
            for keypoint, relevant_knowledge in zip(key_points, relevant_knowledges):
                if not relevant_knowledge:
                    continue
                keypoint_questions = self.generate_keypoint_questions(
                    project, keypoint, relevant_knowledge
                )
                questions["keypoint_questions"].append(
                    dict(
                        keypoint=keypoint,
                        references=relevant_knowledge,
                        questions=keypoint_questions,
                    )
                )

            question_list.append(questions)

        return question_list
