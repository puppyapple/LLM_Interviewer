from icecream import ic
from datetime import datetime
from haystack.components.converters import (
    PDFMinerToDocument,
    DOCXToDocument,
    TextFileToDocument,
)
from pathlib import Path
from loguru import logger
from functools import lru_cache
from .prompts import RESUME_ANALYSIS_SYSTEM_PROMPT, RESUME_PROJECT_EXTRACTION_PROMPT
from .llm_client import get_openai_client


class ResumeParser:
    def __init__(self, api_type="openai"):
        self.model, self.client = get_openai_client(api_type)
        logger.info(f"Using API type: {api_type}, model: {self.model}")

    @lru_cache(maxsize=100)
    def extract_projects(self, resume_text):
        logger.info("提取项目信息...")
        prompt = RESUME_PROJECT_EXTRACTION_PROMPT.format(resume_text=resume_text)

        response = self.client.chat.completions.create(
            model=self.model,
            top_p=1,
            messages=[
                {"role": "system", "content": RESUME_ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        projects = response.choices[0].message.content
        logger.info("项目信息提取完成")
        return projects

    def parse(self, resume_path):
        # 获取文件后缀
        file_extension = Path(resume_path).suffix.lower()

        # 根据文件后缀选择合适的转换器
        if file_extension == ".pdf":
            converter = PDFMinerToDocument()
        elif file_extension in [".docx", ".doc"]:
            converter = DOCXToDocument()
        else:
            converter = TextFileToDocument()

        # 读取并转换文件内容
        doc = converter.run(
            sources=[resume_path], meta={"date_added": datetime.now().isoformat()}
        )

        # 提取文本内容
        text = doc["documents"][0].content
        ic(text, len(text))

        # 使用大模型 API 提取项目信息
        projects_str = self.extract_projects(text)
        if projects_str is None:
            return []
        if getattr(self.extract_projects, "cache_info", None):
            cache_info = self.extract_projects.cache_info()
            if cache_info.hits > 0:
                logger.info(f"使用缓存的项目信息 (命中次数: {cache_info.hits})")
        projects = [t for t in projects_str.split("\n\n") if len(t) > 20]
        return projects
