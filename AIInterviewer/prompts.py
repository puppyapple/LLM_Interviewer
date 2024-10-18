RESUME_ANALYSIS_SYSTEM_PROMPT = "你是一个专业的简历分析助手。"

INTERVIEW_SYSTEM_PROMPT = "你是一个专业的面试官。"

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

请以清晰、结构化的方式呈现这些文本块,使其易于阅读和理解。每个文本块应该是一个独立的段落,包含该经历的所有相关信息。

简历文本：
{resume_text}

每个项目请严格按照上面的关键信息返回结果，不要有多余的输出，也不要有任何遗漏；
如果项目中没有使用到上面的关键信息，请留空。
"""

KEY_POINTS_EXTRACTION_PROMPT = """
请仔细阅读以下项目描述,并提取出其中的技术关键点。每个关键点应该简洁明了,便于后续处理。

项目描述:
{project}

请列出5-10个关键点,每个关键点占一行。
"""

QUESTION_GENERATION_PROMPT = """
基于以下项目描述和相关知识点,生成3-5个针对性强的面试问题。这些问题应该能够深入考察候选人在项目中的实际贡献、技术理解和解决问题的能力。

项目描述:
{project}

相关知识点:
{relevant_knowledge}

生成的问题应该:
1. 直接关联项目中使用的技术或解决的问题
2. 要求候选人解释具体的技术选择或实现细节
3. 探讨候选人在项目中遇到的挑战及其解决方案
4. 评估候选人对相关技术领域的深入理解

请列出生成的问题,每个问题占一行。
"""
