"""
parser.py
文档解析：把 PDF / DOCX / TXT 转成纯文本。

常见坑（重要）：
    PDF:
        - 双栏排版会被按行读取，导致左右栏混在一起
        - 页眉页脚会重复出现
        - 换行符断裂：一行的最后一个字和下一行的第一个字被空格或换行分开
        - 提取出的文本里可能有零宽字符
    DOCX:
        - 表格内容默认不被提取（用 python-docx 需要手动处理）
        - 文本框、页眉页脚要单独处理
    TXT:
        - 编码不确定，可能是 utf-8 / gbk / utf-8-sig

解决方案：
    - PDF: 优先用 pdfplumber（布局感知更好），失败退回 pypdf
    - DOCX: 主文本 + 表格都提取
    - TXT: 自动尝试多种编码
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from utils import setup_logger, clean_text, file_hash

logger = setup_logger()


# ==================================================================
# 数据类
# ==================================================================
@dataclass
class Document:
    """
    解析后的文档。

    source:     文件路径
    text:       完整纯文本
    metadata:   元信息（页数、表格数等）
    file_hash:  文件 MD5，用于去重
    """
    source: str
    text: str
    metadata: dict = field(default_factory=dict)
    file_hash: str = ""

    @property
    def num_chars(self) -> int:
        return len(self.text)

    def __repr__(self):
        return f"Document(source={Path(self.source).name}, chars={self.num_chars})"


# ==================================================================
# PDF 解析
# ==================================================================
def parse_pdf_pdfplumber(path: str) -> str:
    """
    用 pdfplumber 解析 PDF。它是布局感知的，能更好处理双栏和表格。
    但速度比 pypdf 慢。
    """
    import pdfplumber

    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            texts.append(page_text)
    return "\n".join(texts)


def parse_pdf_pypdf(path: str) -> str:
    """用 pypdf 解析 PDF。速度快，但对复杂布局效果一般。"""
    from pypdf import PdfReader

    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)


def parse_pdf(path: str) -> Document:
    """PDF 解析入口。先试 pdfplumber，失败退回 pypdf。"""
    logger.info(f"[PDF] 解析: {Path(path).name}")

    text = ""
    method = ""

    try:
        text = parse_pdf_pdfplumber(path)
        method = "pdfplumber"
    except Exception as e:
        logger.warning(f"[PDF] pdfplumber 失败 ({e})，尝试 pypdf")
        try:
            text = parse_pdf_pypdf(path)
            method = "pypdf"
        except Exception as e2:
            logger.error(f"[PDF] pypdf 也失败: {e2}")
            raise

    text = clean_text(text)
    logger.info(f"[PDF] 解析完成 ({method}): {len(text)} 字符")

    # 统计页数
    try:
        from pypdf import PdfReader
        num_pages = len(PdfReader(path).pages)
    except Exception:
        num_pages = -1

    return Document(
        source=path,
        text=text,
        metadata={"format": "pdf", "method": method, "num_pages": num_pages},
        file_hash=file_hash(path),
    )


# ==================================================================
# DOCX 解析
# ==================================================================
def parse_docx(path: str) -> Document:
    """
    解析 DOCX。

    注意: python-docx 的 document.paragraphs 不包含表格内容，
    需要额外遍历 document.tables。
    """
    from docx import Document as DocxDocument

    logger.info(f"[DOCX] 解析: {Path(path).name}")

    doc = DocxDocument(path)

    parts = []

    # 1. 正文段落
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)

    # 2. 表格内容
    num_tables = len(doc.tables)
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    text = clean_text("\n".join(parts))
    logger.info(f"[DOCX] 解析完成: {len(text)} 字符, {num_tables} 个表格")

    return Document(
        source=path,
        text=text,
        metadata={
            "format": "docx",
            "num_paragraphs": len(doc.paragraphs),
            "num_tables": num_tables,
        },
        file_hash=file_hash(path),
    )


# ==================================================================
# TXT 解析
# ==================================================================
def parse_txt(path: str) -> Document:
    """
    解析 TXT。自动尝试多种编码。
    """
    logger.info(f"[TXT] 解析: {Path(path).name}")

    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "big5", "latin-1"]
    text = None
    used_encoding = ""

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            used_encoding = enc
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise ValueError(f"无法解码文件: {path}")

    text = clean_text(text)
    logger.info(f"[TXT] 解析完成 ({used_encoding}): {len(text)} 字符")

    return Document(
        source=path,
        text=text,
        metadata={"format": "txt", "encoding": used_encoding},
        file_hash=file_hash(path),
    )


# ==================================================================
# 统一入口
# ==================================================================
SUPPORTED_EXTS = {".pdf", ".docx", ".doc", ".txt"}


def parse_document(path: str) -> Optional[Document]:
    """
    统一解析入口。根据扩展名分发。
    不支持的格式返回 None。
    """
    path = str(path)
    ext = Path(path).suffix.lower()

    if not os.path.exists(path):
        logger.error(f"文件不存在: {path}")
        return None

    if ext not in SUPPORTED_EXTS:
        logger.warning(f"不支持的格式: {ext}")
        return None

    try:
        if ext == ".pdf":
            return parse_pdf(path)
        elif ext in (".docx", ".doc"):
            return parse_docx(path)
        elif ext == ".txt":
            return parse_txt(path)
    except Exception as e:
        logger.error(f"解析失败 {path}: {e}")
        return None


def parse_directory(dir_path: str, recursive: bool = True) -> List[Document]:
    """
    批量解析目录下所有支持的文档。
    """
    docs = []
    dir_path = Path(dir_path)

    if not dir_path.exists():
        logger.error(f"目录不存在: {dir_path}")
        return docs

    pattern = "**/*" if recursive else "*"
    for p in dir_path.glob(pattern):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            doc = parse_document(str(p))
            if doc and doc.text:
                docs.append(doc)

    logger.info(f"[Batch] 共解析 {len(docs)} 个文档")
    return docs


# ==================================================================
# 测试
# ==================================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        doc = parse_document(path)
        if doc:
            print(f"\n来源: {doc.source}")
            print(f"格式: {doc.metadata.get('format')}")
            print(f"字符数: {doc.num_chars}")
            print(f"MD5: {doc.file_hash}")
            print(f"\n前 500 字:\n{doc.text[:500]}")
    else:
        print("用法: python parser.py <文件路径>")
        print("\n创建测试文件:")
        print("  echo '这是一个测试文档。它包含多个句子。用于验证解析。' > data/uploads/test.txt")
        print("  python parser.py data/uploads/test.txt")
