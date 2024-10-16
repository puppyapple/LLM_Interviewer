from knowledge_base import KnowledgeBase
from retriever import HybridSearchRetriever
from openai import OpenAI
from .prompts import KEY_POINTS_EXTRACTION_PROMPT, QUESTION_GENERATION_PROMPT


class QuestionGenerator:
    def __init__(self, api_type="openai"):
        self.knowledge_base = KnowledgeBase()
        self.retriever = HybridSearchRetriever(self.knowledge_base.document_store)
        self.client = OpenAI()  # 初始化OpenAI客户端
        self.model = "gpt-3.5-turbo"  # 使用适当的模型

    def generate_questions(self, projects):
        questions = []
        for project in projects:
            # 提取项目的关键点
            key_points = self.extract_key_points(project)

            # 使用混合检索检索相关知识点
            relevant_knowledge = self.retrieve_knowledge(key_points)

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
                {"role": "system", "content": "你是一个专业的项目分析助手。"},
                {
                    "role": "user",
                    "content": KEY_POINTS_EXTRACTION_PROMPT.format(project=project),
                },
            ],
        )
        return response.choices[0].message.content.split("\n")

    def retrieve_knowledge(self, key_points):
        all_relevant_docs = []
        for point in key_points:
            relevant_docs = self.retriever.retrieve(point, top_k=1)
            all_relevant_docs.extend(relevant_docs)
        return "\n".join(all_relevant_docs)

    def generate_project_questions(self, project, relevant_knowledge):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "你是一个专业的面试官。"},
                {
                    "role": "user",
                    "content": QUESTION_GENERATION_PROMPT.format(
                        project=project, relevant_knowledge=relevant_knowledge
                    ),
                },
            ],
        )
        return response.choices[0].message.content.split("\n")
