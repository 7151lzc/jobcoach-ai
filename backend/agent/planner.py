# backend/agent/planner.py
import os
import json
import dashscope
from dashscope import Generation
from dotenv import load_dotenv
from agent.tools import search_company_info, generate_interview_questions, check_salary_range

load_dotenv()

dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# 工具注册表：名字 → 实际函数
# Agent 调用工具时，通过名字在这里找到对应函数执行
TOOL_MAP = {
    "search_company_info": search_company_info,
    "generate_interview_questions": generate_interview_questions,
    "check_salary_range": check_salary_range,
}

# 工具描述：告诉 LLM 有哪些工具可以用
# 这是 OpenAI function calling 的标准格式，通义千问兼容这个格式
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_company_info",
            "description": "当需要了解目标公司的基本信息时使用，包括公司规模、主营业务、技术栈、融资情况。输入公司名称，返回公司概况。",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "公司名称，如'字节跳动'、'阿里巴巴'"
                    }
                },
                "required": ["company_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_interview_questions",
            "description": "根据岗位名称和技能要求生成面试题。当用户想知道面试会被问什么时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_title": {
                        "type": "string",
                        "description": "岗位名称，如'大模型应用开发工程师'"
                    },
                    "skills": {
                        "type": "string",
                        "description": "核心技能，如'RAG, LangChain, 向量数据库'"
                    }
                },
                "required": ["job_title", "skills"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_salary_range",
            "description": "查询某岗位在特定城市的市场薪资范围。当用户想了解薪资水平时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_title": {
                        "type": "string",
                        "description": "岗位名称"
                    },
                    "city": {
                        "type": "string",
                        "description": "城市名称，如'北京'、'上海'、'深圳'"
                    }
                },
                "required": ["job_title", "city"]
            }
        }
    }
]

def run_agent(user_input: str, chat_history: list = []) -> str:
    """
    手动实现 ReAct 循环
    
    为什么手动实现而不用 LangChain AgentExecutor？
    1. 绕开 ChatTongyi 的流式 bug
    2. 能完全控制每一步，调试更方便
    3. 理解原理：Agent 本质就是一个 while 循环
    """
    
    # 构建初始消息列表
    messages = [
        {"role": "system", "content": "你是 JobCoach AI，一位专业的求职顾问。根据用户需求，合理使用工具获取信息，最终给出完整、具体、可执行的建议。"}
    ]
    
    # 加入历史对话
    for msg in chat_history:
        messages.append(msg)
    
    # 加入当前用户输入
    messages.append({"role": "user", "content": user_input})
    
    print(f"\n{'='*50}")
    print(f"用户输入: {user_input}")
    print(f"{'='*50}")
    
    # ReAct 循环，最多跑 5 轮防止无限循环
    for iteration in range(5):
        print(f"\n--- 第 {iteration + 1} 轮思考 ---")
        
        # 调用 LLM，传入工具描述
        response = Generation.call(
            model="qwen-max",
            messages=messages,
            tools=TOOLS_SCHEMA,
            result_format="message"
            # 不传 stream 参数，默认非流式，稳定
        )
        
        assistant_message = response.output.choices[0].message
        finish_reason = response.output.choices[0].finish_reason
        
        print(f"finish_reason: {finish_reason}")
        
        # 把 LLM 的回复加入消息历史
        # 为什么要加？因为下一轮 LLM 需要知道自己上一步做了什么
        messages.append({
            "role": "assistant",
            "content": assistant_message.get("content", ""),
            "tool_calls": assistant_message.get("tool_calls", None)
        })
        
        # 判断 LLM 是否决定调用工具
        if finish_reason == "tool_calls":
            tool_calls = assistant_message.get("tool_calls", [])
            
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                tool_call_id = tool_call["id"]
                
                print(f"调用工具: {tool_name}")
                print(f"参数: {tool_args}")
                
                # 执行工具
                # .invoke() 是 LangChain @tool 装饰器提供的调用方法
                tool_func = TOOL_MAP[tool_name]
                tool_result = tool_func.invoke(tool_args)
                
                print(f"工具结果: {tool_result[:100]}...")
                
                # 把工具结果塞回消息列表
                # role="tool" 是专门给工具返回值用的角色
                # LLM 下一轮看到这个，就知道工具返回了什么
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_result
                })
            
            # 工具执行完，继续下一轮循环让 LLM 思考
            continue
        
        # finish_reason == "stop" 说明 LLM 觉得信息够了，生成最终回答
        if finish_reason == "stop":
            final_answer = assistant_message.get("content", "")
            print(f"\n最终回答生成完毕，长度: {len(final_answer)} 字符")
            return final_answer
    
    # 超过最大循环次数，返回最后一次的内容
    return "抱歉，我需要更多信息才能给出完整建议，请尝试更具体的问题。"