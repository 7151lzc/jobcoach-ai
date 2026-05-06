# JobCoach AI 🎯

> 上传简历 + 输入 JD，AI 自动分析技能差距、生成面试题、给出备考建议

[![演示视频]
(https://www.bilibili.com/video/BV1hbRqB8EXG/?vd_source=94ab7663b6834ded682a4ee02719e259)]


[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 项目简介

JobCoach AI 是一个基于 RAG + Agent 架构的求职助手。用户上传 PDF 简历并输入目标岗位 JD，系统自动完成：

- **技能差距分析**：识别候选人与岗位的匹配度，给出具体改进建议
- **多查询语义检索**：并行召回去重，解决单一查询召回不全问题
- **Agent 多步推理**：自动搜索公司信息、生成面试题、查询薪资范围
- **多轮对话记忆**：支持全量/滑动窗口/摘要压缩三种记忆策略

## 技术架构

```
┌─────────────────────────────────────────┐
│              用户请求                    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│           FastAPI 接口层                 │
│  /upload  /analyze  /agent  /health     │
└──────┬───────────────────┬──────────────┘
       │                   │
┌──────▼──────┐    ┌───────▼──────────────┐
│  RAG 模块   │    │     Agent 模块        │
│             │    │                      │
│ PDF解析     │    │  ReAct 循环           │
│ 文本分块    │    │  ├─ search_company    │
│ 向量检索    │    │  ├─ gen_interview_q   │
│ 多查询召回  │    │  └─ check_salary     │
└─────────────┘    │                      │
                   │  记忆管理             │
                   │  ├─ 全量历史          │
                   │  ├─ 滑动窗口          │
                   │  └─ 摘要压缩          │
                   └──────────────────────┘
```

## 目录结构

```
jobcoach-ai/
├── backend/
│   ├── main.py              # FastAPI 入口
│   ├── memory.py            # 对话记忆管理（三种策略）
│   ├── rag/
│   │   ├── loader.py        # PDF 解析（PyMuPDF）
│   │   ├── chunker.py       # 文本分块，中文分隔符优化
│   │   └── retriever.py     # 向量化、存储、多查询检索
│   └── agent/
│       ├── tools.py         # 工具定义（@tool 装饰器）
│       └── planner.py       # ReAct 循环（DashScope 原生）
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/7151lzc/jobcoach-ai.git
cd jobcoach-ai
```

### 2. 创建虚拟环境

```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# Mac / Linux
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
cp backend/.env.example backend/.env
```

编辑 `backend/.env`：

```env
DASHSCOPE_API_KEY=你的阿里云DashScope Key
```

> 在 [DashScope 控制台](https://dashscope.aliyun.com) 注册获取免费额度

### 5. 启动服务

```bash
cd backend
uvicorn main:app --reload --port 8000
```

访问 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 打开交互式 API 文档。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/upload` | 上传简历 PDF，触发解析和向量化 |
| POST | `/analyze` | RAG 分析简历与 JD 匹配度 |
| POST | `/agent` | Agent 多步推理，自动调用工具 |
| GET | `/memory/status` | 查看当前记忆状态 |
| POST | `/memory/clear` | 清空对话，开始新会话 |
| GET | `/health` | 健康检查 |

### 示例：多轮对话

第一轮：
```bash
curl -X POST "http://127.0.0.1:8000/agent" \
  -H "Content-Type: application/json" \
  -d '{"message": "我要应聘字节跳动的大模型开发岗位，帮我了解这家公司"}'
```

第二轮（无需重复上下文）：
```bash
curl -X POST "http://127.0.0.1:8000/agent" \
  -H "Content-Type: application/json" \
  -d '{"message": "那这个岗位薪资大概多少？"}'
```

## 核心设计决策

**为什么用多查询检索而不是单一查询？**

单一查询因语义角度单一容易召回不全。实现三路并行检索再去重合并，技能覆盖率从约 60% 提升至 90%+。这是在实际测试中发现并解决的真实问题。

**为什么手动实现 ReAct 循环而不用 LangChain AgentExecutor？**

`langchain_community` 的 `ChatTongyi` 在 Agent 场景下存在流式 tool_calls 的 IndexError bug。手动实现 ReAct 循环绕开了这个问题，同时对每一步有完全控制权。Agent 本质是一个带工具调用的 while 循环。

**记忆策略为什么选滑动窗口？**

全量历史最准确但 token 成本线性增长；滑动窗口（最近 6 条）平衡成本和效果；摘要压缩最优但需要额外 LLM 调用。策略可通过配置切换。

## 踩过的坑

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| RAG 召回不全 | k=3 太小，相关 chunk 排在第4位被截断 | 多查询并行检索，从3个角度召回再去重 |
| ChatTongyi Agent 报错 | 流式模式下 tool_calls 索引越界 bug | 改用 DashScope 原生 SDK 手动实现 ReAct |
| chunk 含垃圾内容 | PDF 解析出页码、页眉等噪音 | 过滤长度小于30字符的 chunk |
| 中文切块语义断裂 | 默认分隔符只有英文标点 | 添加中文标点到 separators |

## 完成进度

- [x] RAG 核心模块：PDF 解析 + 向量检索 + FastAPI 接口
- [x] 多查询召回优化
- [x] Agent 多步推理：ReAct 循环 + 三个工具
- [x] 多轮对话记忆：三种策略实现
- [ ] 流式输出（SSE）：分析结果实时渲染
- [ ] 用户系统：JWT 鉴权 + 历史记录
- [ ] 前端界面：React + Tailwind

## 环境要求

- Python 3.11+
- DashScope API Key（[免费注册](https://dashscope.aliyun.com)）

## License

MIT © [7151lzc](https://github.com/7151lzc)
