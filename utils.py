"""
utils.py
通用工具：日志、文本清洗、文件哈希。
"""

import hashlib
import re
from pathlib import Path
from loguru import logger


def setup_logger(log_file: str = None):
    """配置 loguru 日志。"""
    logger.remove()   # 移除默认 handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<level>{message}</level>",
    )
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level="DEBUG", rotation="10 MB")
    return logger


def clean_text(text: str) -> str:
    """
    基础文本清洗：
        - 统一换行符
        - 去掉多余空白（保留单个换行）
        - 去掉零宽字符
    """
    if not text:
        return ""

    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 去掉零宽字符（PDF 提取常见）
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # 全角空格转半角
    text = text.replace("\u3000", " ")
    # 连续空白折成一个空格
    text = re.sub(r"[ \t]+", " ", text)
    # 3 个以上换行折成 2 个
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 每行首尾空格
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def normalize_for_match(text: str) -> str:
    """
    用于相似度匹配的强归一化：
        - 去掉所有空白
        - 去掉标点
        - 转小写
    这样"你好，世界！" 和 "你好 世界" 会被视为相同。
    """
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    # 中英文标点
    _PUNCT = "，。！？、；：""''（）《》【】,.!?;:\"'()[]<>—…·"
    text = re.sub("[" + re.escape(_PUNCT) + "]", "", text)
    return text.lower()


def file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """计算文件 MD5，用于知识库去重。"""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def text_hash(text: str) -> str:
    """计算文本 MD5。"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def truncate(text: str, max_len: int = 100) -> str:
    """截断文本用于日志显示。"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


if __name__ == "__main__":
    setup_logger()
    logger.info("测试日志")
    logger.info(f"清洗测试: {clean_text('你好   世界\n\n\n\n再见')!r}")
    logger.info(f"归一化测试: {normalize_for_match('你好，世界！')!r}")
