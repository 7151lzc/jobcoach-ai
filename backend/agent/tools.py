# backend/agent/tools.py
import os
from langchain.tools import tool
from langchain_community.chat_models import ChatTongyi
from dotenv import load_dotenv

load_dotenv()

# @tool 装饰器：把一个普通函数变成 LLM 可以调用的工具
# 函数的 docstring 就是 description，LLM 靠它决定什么时候调用
# 这是 LangChain 最优雅的设计之一

@tool
def search_company_info(company_name: str) -> str:
    """
    当需要了解目标公司的基本信息时使用，包括：公司规模、
    主营业务、技术栈、融资情况、企业文化。
    输入公司名称，返回公司概况。
    """
    # 现阶段用 LLM 模拟搜索，第二阶段可以接入真实搜索API
    # 为什么先用模拟？因为让 Agent 流程跑通比接真实API更重要
    # 真实API（比如Tavily、Serper）接入只需要换这一个函数
    llm = ChatTongyi(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-max",
        streaming=False
    )
    response = llm.invoke(
        f"请简要介绍{company_name}这家公司：主营业务、规模、技术栈、企业文化，100字以内。"
    )
    return response.content

@tool
def generate_interview_questions(job_title: str, skills: str) -> str:
    """
    根据岗位名称和技能要求生成面试题。
    当用户想知道面试会被问什么问题时使用。
    输入岗位名称和核心技能，返回5个高频面试题及回答思路。
    """
    llm = ChatTongyi(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-max",
        streaming=False
    )
    response = llm.invoke(
        f"为{job_title}岗位生成5个面试题，重点考察{skills}。"
        f"每题给出简短回答思路，格式：Q：问题\\nA：思路\\n"
    )
    return response.content

@tool
def check_salary_range(job_title: str, city: str) -> str:
    """
    查询某岗位在特定城市的市场薪资范围。
    当用户想了解薪资水平或判断offer是否合理时使用。
    输入岗位名称和城市，返回薪资范围和影响因素。
    """
    llm = ChatTongyi(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-max",
        streaming=False
    )
    response = llm.invoke(
        f"{city}的{job_title}岗位薪资范围是多少？"
        f"分应届生/1-3年/3年以上分别说明，100字以内。"
    )
    return response.content

# 把所有工具放进列表，方便 Agent 使用
JOBCOACH_TOOLS = [
    search_company_info,
    generate_interview_questions,
    check_salary_range,
]