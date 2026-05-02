import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

embedding_model = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

def build_vectorstore(chunks: list[str]) -> Chroma:
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    return vectorstore

def load_vectorstore() -> Chroma:
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )

def retrieve(query: str, vectorstore: Chroma, k: int = 3) -> list[str]:
    """方案一：直接加大k值"""
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

def retrieve_with_score(query: str, vectorstore: Chroma, k: int = 8, threshold: float = 0.3) -> list[str]:
    """
    方案二：带相似度分数的检索，过滤掉低质量结果
    
    先取k=8个候选，再过滤掉相似度低于threshold的
    这样既扩大了召回范围，又不会把完全不相关的内容塞进prompt
    """
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    
    # results是[(Document, score), ...]的列表
    # score越高越相关，范围0到1
    filtered = [
        doc.page_content 
        for doc, score in results 
        if score >= threshold
    ]
    
    # 如果过滤后一条都没有，退回到直接取前3条，防止返回空结果
    if not filtered:
        return [doc.page_content for doc, _ in results[:3]]
    
    return filtered

def retrieve_with_rerank(query: str, vectorstore: Chroma, k: int = 10) -> list[str]:
    """
    方案三：多查询rerank，从不同角度检索再合并去重
    
    一个问题可以拆成多个子查询，分别检索再合并
    解决"单一查询角度导致遗漏"的问题
    """
    # 把原始查询拆成多个角度
    sub_queries = [
        query,
        "技术技能 编程语言 框架工具",   # 从技能角度补充检索
        "项目经验 工作经历 实战",        # 从经验角度补充检索
    ]
    
    seen = set()
    all_chunks = []
    
    for sub_query in sub_queries:
        results = vectorstore.similarity_search(sub_query, k=4)
        for doc in results:
            # 用内容的前50个字符作为去重key，避免重复chunk进入结果
            key = doc.page_content[:50]
            if key not in seen:
                seen.add(key)
                all_chunks.append(doc.page_content)
    
    # 最多返回k个，防止prompt太长
    return all_chunks[:k]