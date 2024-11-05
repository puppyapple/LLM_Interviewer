[上一期内容](https://mp.weixin.qq.com/s/GdNS5QHRsSHdVHM9z9lqdg)匆忙地介绍了一下让大模型解析简历并提取项目关键信息的操作，这期续上更加关键的部分——基于项目关键信息生成面试问题。

说起来挺有意思，大多时候我们都是让大模型来解答问题，而在`AI`面试官的场景下，我们却需要大模型来**生成问题**。

**幻觉问题**是大模型应用中最让人头疼的问题之一。

其实一些大尺寸的模型（尤其是闭源商业模型），在一些相对主流的知识性的问题上，幻觉情况还是相对较少的。

但如果是小尺寸的开源模型，表现就不尽人意了。

为了避免使用开源模型时的效果不佳，这里对问题的生成我也采用了外挂`RAG`的方式。

[更早的一篇内容](https://mp.weixin.qq.com/s/meuW5qrf-dcu_BasP7modg)中，我已经介绍了构建`RAG`知识库的过程（采用了`BM25`和`Embeddings`结合的混合检索方式），感兴趣的可以点击链接回顾一下。

我设想的**面试问题生成**思路大概是这样的：

1. 解析简历，提取项目关键信息（已经在上一期内容中完成了）
2. 对每一个项目文本，抽取其中的关键点，包括：
   - 技术概念
   - 工具
   - 理论框架
   - 方法论
3. 对于每个项目生成问题，包含两种情况：
   - 项目粒度的问题（当项目文本中没有关键点时，直接基于项目文本生成问题）
   - 知识点粒度的问题（当项目文本中包含关键点时，通过`RAG`知识库召回相关的知识点，然后基于关键点和知识点生成问题）

关键点抽取用到的`prompt`如下：

```python
KEY_POINTS_EXTRACTION_PROMPT = """
请仔细阅读以下项目描述,并按照以下要求提取关键技术概念:

1. 识别项目中涉及的具体技术概念、工具、框架或方法论。
2. 为每个识别出的技术概念提供一个简洁的描述(不超过15个字)。
3. 关注那些对项目成功至关重要或体现技术难度的概念。
4. 尽量避免过于宽泛或通用的概念。

项目描述:
{project}

请列出2-3个关键技术概念,每个概念占一行,格式如下:
技术概念: 简洁描述

示例:
Docker: 容器化部署工具
TensorFlow: 深度学习框架
CI/CD: 持续集成和部署
```

问题生成用到的两个`prompt`如下：

```python
PROJECT_QUESTION_GENERATION_PROMPT = """
基于以下项目描述，生成2-3个针对性强的面试问题。这些问题应该能够深入考察候选人在项目中的实际贡献、技术理解和解决问题的能力。

项目描述:
{project}

生成的问题应该:
1. 直接关联项目中使用的技术或解决的问题
2. 要求候选人解释具体的技术选择或实现细节
3. 探讨候选人在项目中遇到的挑战及其解决方案
4. 评估候选人对相关技术领域的深入理解

请列出生成的问题,每个问题占一行。
"""

KEYPOINT_QUESTION_GENERATION_PROMPT = """
基于以下关键技术概念和检索到的相关知识点,生成1-2个针对性强的面试问题。
这些问题应该深入考察候选人对关键技术概念的理解,并且必须基于检索到的相关知识内容。

关键技术概念:
{keypoint}

检索到的相关知识:
{relevant_knowledge}

生成的问题应该:
1. 直接关联关键技术概念,并利用检索到的知识点
2. 要求候选人解释具体的概念原理、实现细节或应用场景
3. 探讨候选人对概念的深入理解,包括优缺点、最佳实践或常见问题
4. 可以包含实际应用案例或特定场景下的问题解决

请列出生成的问题,每个问题占一行。确保每个问题都紧密结合检索到的相关知识,体现出对技术细节的深入询问。

示例:
关键技术概念: Docker容器化
检索到的相关知识: Docker使用轻量级的容器来运行应用,具有快速部署、资源隔离、版本控制等优势。Docker镜像采用分层存储技术,可以共享基础层,节省存储空间。

生成的问题:
1. 请解释Docker的分层存储技术是如何工作的,它如何帮助优化存储空间使用?
2. 在使用Docker进行微服务部署时,您如何处理服务之间的网络通信和数据持久化问题?
3. Docker容器与传统虚拟机相比,在资源利用率和性能方面有什么优势?请举例说明。
"""
```
上面的`prompt`依然都是让`AI`辅助生成的，然后我做了一些调整。
      
> 这里强烈安利一下`Cursor`这款`AI`编程`IDE`，真的非常值得入坑。
>
> 后面甚至打算写一些系列文章介绍自己的使用体验。

言归正传，我让`AI`帮我生成了一份大模型从业者的简历，然后基于这份简历生成了面试问题。

其中一个结果样例如下：

```json
{
    "project": "#### 项目一: 多语言翻译模型优化\n- 利用transformer架构和多任务学习,提高了模型在低资源语言上的翻译质量\n- 实现了动态批处理和混合精度训练,将训练速度提升了40%\n- 应用知识蒸馏技术,将模型大小压缩50%,同时保持95%的性能",
    "project_questions": [
        "1. 在多语言翻译模型优化项目中，您是如何利用多任务学习来提高低资源语言的翻译质量的？请详细解释您的方法和实现过程。",
        "2. 您提到实现了动态批处理和混合精度训练，能否具体说明这两种技术是如何协同工作以提升训练速度的？",
        "3. 在应用知识蒸馏技术压缩模型大小的过程中，您遇到了哪些主要挑战？您是如何解决这些挑战并确保模型性能的？",
        "4. 在多语言翻译模型优化项目中，您如何评估和验证模型在不同语言上的性能？请分享您的具体做法。",
        "5. 变换器（Transformer）架构在多语言翻译任务中有哪些优势和局限性？您在项目中是如何应对这些局限性的？"
    ],
    "keypoint_questions": [
        {
            "keypoint": "Transformer: 基于自注意力机制的模型架构  ",
            "references": "**1.5 在常规attention中，一般有k=v，那self-attention 可以吗?** self-attention实际只是attention中的一种特殊情况，因此k=v是没有问题的，也即K，V参数矩阵相同。实际上，在Transformer模型中，Self-Attention的典型实现就是k等于v的情况。Transformer中的Self-Attention被称为\"Scaled Dot-Product Attention\"，其中通过将词向量进行线性变换来得到Q、K、V，并且这三者是相等的。",
            "questions": [
                "1. 在Transformer模型中，Self-Attention是如何实现的？请详细解释Scaled Dot-Product Attention的过程，特别是Q、K、V的生成和计算方式。",
                "2. 为什么在Self-Attention机制中，K和V通常设置为相同的矩阵？这种设置对模型的性能和效果有哪些影响？",
                "3. Transformer模型中的Self-Attention机制有哪些优点和缺点？在实际应用中，如何克服这些缺点？",
                "4. 在一个具体的自然语言处理任务中，例如机器翻译，Self-Attention机制是如何帮助模型更好地捕捉长距离依赖关系的？请结合实际案例进行说明。"
            ]
        },
        {
            "keypoint": "多任务学习: 同时训练多个相关任务  ",
            "references": "3.  **增量学习（Incremental Learning）**：将微调过程分为多个阶段，每个阶段只微调一小部分参数。这样可以逐步引入新任务，减少参数更新的冲突，降低灾难性遗忘的风险。 4.  **多任务学习（Multi-Task Learning）**：在微调过程中，同时训练多个相关任务，以提高模型的泛化能力和抗遗忘能力。通过共享模型参数，可以在不同任务之间传递知识，减少灾难性遗忘的影响。 综上所述，灾难性遗忘是在模型微调过程中可能出现的问题。通过合适的方法和技术，可以减少灾难性遗忘的发生，保留之前学习到的知识，提高模型的整体性能。",
            "questions": [
                "1. 请解释多任务学习（Multi-Task Learning）的基本原理，以及它是如何通过共享模型参数在不同任务之间传递知识的？",
                "2. 在多任务学习中，如何平衡不同任务之间的损失函数，以确保所有任务都能得到有效的训练？请提供一个具体的例子。",
                "3. 结合增量学习（Incremental Learning）和多任务学习，如何逐步引入新任务以减少参数更新的冲突，并降低灾难性遗忘的风险？请详细描述这一过程。",
                "4. 在实际应用中，多任务学习如何提高模型的泛化能力和抗遗忘能力？请举一个具体的场景，并说明其优势和潜在挑战。"
            ]
        },
        {
            "keypoint": "知识蒸馏: 模型压缩技术",
            "references": "什么是知识 ？ 这里知识指的是**模型的参数本身**，本质是把模型从输入映射到输出的过程。知识蒸馏就是想把这种映射能力从大模型迁移到小模型上。 !",
            "questions": [
                "1. 请解释知识蒸馏中“模型的参数本身”指的是什么，以及它是如何实现从大模型到小模型的知识迁移的？",
                "2. 在知识蒸馏过程中，有哪些具体的实现细节和技术手段可以确保小模型能够有效学习到大模型的映射能力？",
                "3. 知识蒸馏在实际应用中有哪些常见的问题和挑战？您如何解决这些问题以确保模型压缩后的性能不显著下降？",
                "4. 请描述一个实际的应用场景，说明如何利用知识蒸馏技术将一个复杂的深度学习模型压缩为一个轻量级模型，并讨论其在资源受限设备上的表现。"
            ]
        }
    ]
}
```
上面的问题生成使用的是`Ollama`支持的`Qwen2.5:14b(int4)`模型。

可以看到效果还是不错的，当然，如果想要进一步提升效果，有条件的话可以考虑使用`Qwen-Plus/Max、GPT-4o`之类的闭源模型。

整个**问题生成器**通过一个`QuestionGenerator`类来完成，完整的代码如下：

```python
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

```
