from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    
    # 过滤掉太短的块
    chunks = [c.strip() for c in chunks if len(c.strip()) > 30]
    
    return chunks