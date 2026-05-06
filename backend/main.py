import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.chat_models import ChatTongyi
from langchain.schema import HumanMessage, SystemMessage,AIMessage
from dotenv import load_dotenv

from rag.loader import load_pdf
from rag.chunker import chunk_text
from rag.retriever import build_vectorstore, load_vectorstore, retrieve_with_rerank

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatTongyi(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-max"
)

vectorstore = None

class AnalyzeRequest(BaseModel):
    jd_text: str
    question: str

@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    global vectorstore

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    text = load_pdf(tmp_path)
    chunks = chunk_text(text)
    vectorstore = build_vectorstore(chunks)
    os.unlink(tmp_path)

    return {"status": "ok", "chunks": len(chunks)}

@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    if vectorstore is None:
        return {"error": "请先上传简历"}

    # 从retrieve改成retrieve_with_rerank
    resume_chunks = retrieve_with_rerank(request.question, vectorstore, k=10)
    context = "\n\n".join(resume_chunks)

    messages = [
        SystemMessage(content="你是一位资深HR顾问，擅长分析候选人和岗位的匹配度。回答要具体、可执行。"),
        HumanMessage(content=f"""请根据以下信息分析候选人匹配度：

【简历相关内容】
{context}

【目标岗位JD】
{request.jd_text}

【问题】
{request.question}

请指出匹配的技能，以及3个主要技能差距和具体改进建议。""")
    ]

    response = llm.invoke(messages)
    return {"analysis": response.content}

@app.get("/health")
def health():
    return {"status": "ok"}


# 在 main.py 顶部 import 区域加上
from agent.planner import run_agent

# 新增请求模型
class AgentRequest(BaseModel):
    message: str
    chat_history: list = []

@app.post("/agent")
async def agent_chat(request: AgentRequest):
    # 把用户输入存入记忆
    memory.add("user", request.message)
    
    # 取出上下文（根据策略自动处理）
    context = memory.get_context()
    
    # 去掉最后一条（当前用户输入），因为 run_agent 会自己加
    history = context[:-1]
    
    result = run_agent(request.message, history)
    
    # 把 AI 回复也存入记忆
    memory.add("assistant", result)
    
    return {
        "reply": result,
        "turn": memory.turn_count  # 顺便返回当前是第几轮，方便调试
    }

@app.post("/memory/clear")
def clear_memory():
    """清空对话记忆，开始新会话"""
    memory.clear()
    return {"status": "ok", "message": "记忆已清空"}

@app.get("/memory/status")
def memory_status():
    """查看当前记忆状态，调试用"""
    return {
        "strategy": memory.strategy,
        "total_messages": len(memory.messages),
        "turns": memory.turn_count,
        "context_size": len(memory.get_context())
    }


from memory import ConversationMemory
# 每个会话一个记忆对象
# 现在先用单个全局对象，后面用户系统做好后改成 dict
memory = ConversationMemory(strategy="window", window_size=6)





# 在 main.py 顶部加这两个 import
from fastapi.responses import StreamingResponse
import json

# 新增流式分析接口
@app.post("/analyze/stream")
async def analyze_stream(request: AnalyzeRequest):
    """
    流式版本的 /analyze
    返回 SSE 格式的数据流，前端逐字渲染
    """
    if vectorstore is None:
        return {"error": "请先上传简历"}

    resume_chunks = retrieve_with_rerank(request.question, vectorstore, k=10)
    context = "\n\n".join(resume_chunks)

    prompt = f"""请根据以下信息分析候选人匹配度：

【简历相关内容】
{context}

【目标岗位JD】
{request.jd_text}

【问题】
{request.question}

请指出匹配的技能，以及3个主要技能差距和具体改进建议。"""

    async def generate():
        import dashscope
        from dashscope import Generation
        import json
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

        # 用队列在线程和异步之间传递数据
        queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def call_sync():
            """在独立线程里运行同步的流式调用"""
            try:
                responses = Generation.call(
                    model="qwen-max",
                    messages=[
                        {"role": "system", "content": "你是专业的求职顾问，回答要具体可执行。"},
                        {"role": "user", "content": prompt}
                    ],
                    stream=True,
                    incremental_output=True,
                    result_format="message"
                )
                for response in responses:
                    if response.status_code == 200:
                        choices = getattr(response.output, "choices", None)
                        if choices:
                            content = getattr(choices[0].message, "content", "")
                            if content:
                                # 把 token 放入队列，异步生成器来取
                                loop.call_soon_threadsafe(queue.put_nowait, content)
                    else:
                        loop.call_soon_threadsafe(queue.put_nowait, None)
                        return
            except Exception as e:
                print(f"线程异常: {e}")
            finally:
                # 发送结束信号
                loop.call_soon_threadsafe(queue.put_nowait, None)

        # 在线程池里运行同步调用，不阻塞事件循环
        executor = ThreadPoolExecutor(max_workers=1)
        loop.run_in_executor(executor, call_sync)

        # 从队列里取 token，yield 给前端
        while True:
            token = await queue.get()
            if token is None:
                break
            yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'done': True})}\n\n"
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",  # SSE 的固定 media_type
        headers={
            "Cache-Control": "no-cache",      # 不缓存，确保实时性
            "X-Accel-Buffering": "no"         # 禁止 nginx 缓冲，直接推给客户端
        }
    )