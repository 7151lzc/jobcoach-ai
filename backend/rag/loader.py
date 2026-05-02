import fitz
import os

def load_pdf(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    doc = fitz.open(file_path)
    full_text = ""
    
    for page in doc:
        full_text += page.get_text("text")
    
    doc.close()
    return full_text.strip()