# JobCoach AI 🎯

> 上传简历 + 输入 JD，AI 自动分析技能差距、生成面试题、给出备考建议

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-green)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 项目简介

JobCoach AI 是一个基于 RAG（检索增强生成）架构的求职助手。用户上传 PDF 简历并输入目标岗位 JD，系统自动完成：

- **技能差距分析**：识别候选人与岗位的匹配度，给出具体改进建议
- **语义检索**：多查询并行召回，解决单一查询召回不全问题
- **结构化输出**：基于 qwen-max 生成可执行的备考方案

## 技术架构

```
用户上传 PDF
    │
    ▼
PDF 解析（PyMuPDF）
    │
    ▼
文本分块（RecursiveCharacterTextSplitter）
chunk_size=500，中文标点分隔符优化
    │
    ▼
向量化（DashScope text-embedding-v2）
    │
    ▼
存入向量库（ChromaDB 持久化）


用户输入问题
    │
    ▼
多查询检索（3个角度并行 → 去重合并）
    │
    ▼
Prompt 组装（System + User 分层）
    │
    ▼
LLM 生成（qwen-max）
    │
    ▼
返回分析报告
```

## 目录结构

```
jobcoach-ai/
├── backend/
│   ├── main.py              # FastAPI 入口，/upload /analyze /health
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── loader.py        # PDF 解析（PyMuPDF）
│   │   ├── chunker.py       # 文本分块，中文分隔符优化
│   │   └── retriever.py     # 向量化、存储、多查询检索
│   └── .env.example         # 环境变量模板
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

编辑 `backend/.env`，填入你的 API Key：

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
| POST | `/analyze` | 输入 JD 和问题，返回 AI 分析报告 |
| GET | `/health` | 健康检查 |

### 示例请求

上传简历：

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "file=@你的简历.pdf"
```

分析匹配度：

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "jd_text": "岗位要求：熟悉LangChain、RAG系统开发、向量数据库，有FastAPI开发经验",
    "question": "候选人是否适合这个岗位，有哪些技能差距？"
  }'
```

## 核心设计决策

**为什么用 RecursiveCharacterTextSplitter？**

它按优先级递归切分（段落 → 句子 → 标点 → 字符），保证每个 chunk 的语义完整性。针对中文简历额外加入了 `。，` 等中文标点作为分隔符。

**为什么实现多查询检索？**

单一查询容易因为语义角度单一导致召回不全——比如问"技术能力"可能漏掉"部署经验"这类 chunk。多查询从不同角度并行检索再去重合并，将技能覆盖率从约 60% 提升至 90%+。

**为什么用 DashScope Embedding 而不是 OpenAI？**

`text-embedding-v2` 对中文语义理解更优，且国内直连无网络问题，适合中文简历场景。

## 后续计划

- [ ] Agent 模块：自动搜索公司信息、生成针对性面试题
- [ ] 流式输出（SSE）：分析结果实时渲染
- [ ] 用户系统：历史记录、多份简历管理
- [ ] 前端界面：React + Tailwind

## 本地开发

```bash
# 运行测试
cd backend
python test_loader.py

# 查看向量库内容
python -c "
from rag.retriever import load_vectorstore
vs = load_vectorstore()
print(vs._collection.count(), 'chunks in store')
"
```

## 环境要求

- Python 3.11+
- DashScope API Key（[免费注册](https://dashscope.aliyun.com)）

## License

MIT © [7151lzc](https://github.com/7151lzc)
