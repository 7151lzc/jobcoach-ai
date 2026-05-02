from rag.loader import load_pdf

# 换成你电脑上任意一个 PDF 的路径
text = load_pdf("D:\\Desktop\\大专实习\\罗志丞个人简历.pdf")
print(f"提取了 {len(text)} 个字符")
print("前200字：")
print(text[:200])


from rag.loader import load_pdf
from rag.chunker import chunk_text

text = load_pdf("D:\\Desktop\\大专实习\\罗志丞个人简历.pdf")
chunks = chunk_text(text)

print(f"总共切出 {len(chunks)} 个块")
print(f"平均每块 {sum(len(c) for c in chunks) // len(chunks)} 字符")
print("\n前3个块：")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- 块 {i+1} ({len(chunk)}字符) ---")
    print(chunk)


from rag.retriever import build_vectorstore, retrieve

# 构建向量库
print("正在构建向量库（会调用 OpenAI API）...")
vectorstore = build_vectorstore(chunks)
print("向量库构建完成")

# 测试检索
query = "候选人有哪些编程技能"
results = retrieve(query, vectorstore, k=3)

print(f"\n查询：{query}")
print(f"检索到 {len(results)} 个相关块：")
for i, r in enumerate(results):
    print(f"\n--- 结果 {i+1} ---")
    print(r)