距离「从零手搓AI面试官」系列的上一次更新稍微有点久了，今天我们继续肝主线剧情。

传统的`RAG`构建也完成了，`LightRAG`也玩过了，现在开始正式构建主角`AIInterviewer`。

大家都面试过，甚至也做过面试官，流程肯定是再熟悉不过了。

作为一名合格的**面试官**，认真「**读简历**」是基本素养，也是对求职者最起码的尊重。

接下来让我们赋予`AIInterviewer`读简历的能力。
        
话不多说还是先上代码：


```python
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
        projects = self.extract_projects(text)
        if getattr(self.extract_projects, "cache_info", None):
            cache_info = self.extract_projects.cache_info()
            if cache_info.hits > 0:
                logger.info(f"使用缓存的项目信息 (命中次数: {cache_info.hits})")

        return projects
```

简历`Parser`本身写的很简单，借助`Haystack`的`haystack.components.converters`组件，可以方便的读取不同格式的简历，目前这里先简单支持`pdf`、`doc/docx`、和纯文本三种格式。

这一步的核心目标是希望`AIInterviewer`能够读懂简历，并从中提取出项目信息。
        
根据我自己写简历的格式偏好和平时看简历的经验，大多数简历都分了「**工作经历**」和「**项目经历**」两块，后者是前者的细化。

所以这里我提取项目信息的方式是：
1. 识别所有的工作经历和项目经历。
2. 分析每段项目经历和工作经历之间的关联性,将相关的项目经历描述整合工作经历一起。
3. 为每个整合后的经历创建一个文本块

下面是我用到的`prompt`：


```python
RESUME_ANALYSIS_SYSTEM_PROMPT = "你是一个专业的简历分析助手。"
RESUME_PROJECT_EXTRACTION_PROMPT = """
请仔细阅读以下简历文本,并按照以下步骤提取和整合相关信息:

1. 识别所有的工作经历和项目经历。

2. 分析每段项目经历和工作经历之间的关联性,将相关的项目经历描述整合工作经历一起。

3. 为每个整合后的经历创建一个文本块,包含以下信息:
   - 时间段
   - 公司/组织名称(如适用)
   - 职位/角色(如适用)
   - 主要职责和成就
   - 相关的项目详情(如适用)
   - 使用的技能和工具

4. 按时间顺序排列这些文本块,最近的经历放在前面。

5. 确保每个文本块的信息完整、连贯,并突出显示候选人的主要成就和技能。

6. 如果某些项目经历与特定的工作经历没有明确关联,可以单独列出。

请以清晰、结构化的方式呈现这些文本块,使其易于阅读和理解。
每个文本块应该是一个独立的段落,包含该经历的所有相关信息。

简历文本：
{resume_text}

每个项目请严格按照上面的关键信息返回结果，不要有多余的输出，也不要有任何遗漏；
如果项目中没有使用到上面的关键信息，请留空。
"""
```

不瞒大家说，这个`prompt`是让`AI`自己写的，写完之后试了试效果还不错。

我对比了下`Qwen2.5:14b`和`Qwen Plus`，开源模型的效果自然是略逊一筹。

考虑到这一步涉及到上下文比较长，且需要复杂的关联推理，所以建议在这一步使用尺寸尽可能大的模型，甚至是商用的`API`。

但这个解析是一次性的，所以可以考虑将解析结果缓存起来，这样下次再解析相同简历的时候，速度就会非常快，开销也会小很多。（见我上方的代码实现）
        
这里之所以把工作和相关的项目经历各自合并到一起，是为了让提取的简历信息在工作经历粒度上聚合得更完整，方便后续的面试（问题生成）阶段。
        
有了分块之后的简历信息，基于此来生成面试问题就会方便很多。

时间有限，这一期来不及介绍这部分内容，下一期再和大家分享啦！


