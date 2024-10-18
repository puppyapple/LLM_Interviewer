from .knowledge_base import KnowledgeBase
from .prompts import (
    KEY_POINTS_EXTRACTION_PROMPT,
    QUESTION_GENERATION_PROMPT,
    INTERVIEW_SYSTEM_PROMPT,
    RESUME_ANALYSIS_SYSTEM_PROMPT,
)
from .llm_client import get_openai_client


class QuestionGenerator:
    def __init__(self, kb: KnowledgeBase, api_type="ollama"):
        self.knowledge_base = kb
        self.model, self.client = get_openai_client(api_type)  # 初始化OpenAI客户端

    def generate_questions(self, projects):
        questions = []
        for project in projects:
            # 提取项目的关键点
            key_points = self.extract_key_points(project)

            # 使用混合检索检索相关知识点
            relevant_knowledge = self.retrieve_knowledge(project)

            # 生成面试问题
            project_questions = self.generate_project_questions(
                project, relevant_knowledge
            )
            questions.extend(project_questions)

        return questions

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

    def retrieve_knowledge(self, projects):
        project_knowledge_pairs = []
        for project in projects:
            relevant_docs = self.knowledge_base.query(project).source_nodes
            if not relevant_docs:
                continue
            knowledge = ""
            for node in relevant_docs:
                context = node.node.metadata["window"]
                knowledge += context + "\n"
            project_knowledge_pairs.append((project, knowledge))
        return project_knowledge_pairs

    def generate_project_questions(self, project, relevant_knowledge):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": INTERVIEW_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": QUESTION_GENERATION_PROMPT.format(
                        project=project, relevant_knowledge=relevant_knowledge
                    ),
                },
            ],
        )
        return response.choices[0].message.content.split("\n")
