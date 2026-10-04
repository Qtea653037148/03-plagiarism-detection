"""
chunker.py
把长文档切成小块，供后续编码和检索。

三种策略：
    sentence  按句切分（最细粒度，适合精确定位抄袭位置）
    paragraph 按段切分（保留语义完整性）
    sliding   滑窗（相邻 N 句作为一个块，保留上下文）

为什么需要分块：
    - 模型有最大长度限制（max_seq_len=128）
    - 长文本向量会丢失细节
    - 抄袭通常是局部片段级别的
"""

import re
from dataclasses import dataclass, field
from typing import List

from utils import setup_logger, text_hash

logger = setup_logger()


# ==================================================================
# 数据类
# ==================================================================
@dataclass
class Chunk:
    """
    一个文本块。

    text:        块的文本
    source:      来源文档路径
    chunk_id:    在文档内的序号
    char_start:  在原文中的起始字符位置
    char_end:    结束位置
    metadata:    额外信息
    """
    text: str
    source: str
    chunk_id: int
    char_start: int = 0
    char_end: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def hash(self) -> str:
        return text_hash(self.text)

    def __repr__(self):
        preview = self.text[:40].replace("\n", " ")
        return f"Chunk[{self.chunk_id}] {preview}..."


# ==================================================================
# 句子切分
# ==================================================================
# 中文断句：句号、问号、感叹号、分号、省略号
# 注意要处理引号内的句号（"他说：'你好。'"）
_SENT_END = re.compile(r"[。！？；!?;]|\.{3,}|…{2,}")


def split_sentences(text: str, min_len: int = 5) -> List[tuple]:
    """
    按标点切句。返回 [(句子文本, 起始位置), ...]。

    处理要点：
        - 句末标点要保留（它是句子的一部分）
        - 过滤太短的片段（往往是标点或空白）
        - 记录每句在原文中的位置，供后续高亮
    """
    sentences = []
    start = 0

    for match in _SENT_END.finditer(text):
        end = match.end()
        sent = text[start:end].strip()
        if len(sent) >= min_len:
            # 记录在原文中的位置（去掉首尾空格后要修正）
            real_start = start + (len(text[start:end]) - len(text[start:end].lstrip()))
            sentences.append((sent, real_start))
        start = end

    # 处理最后一段（没有结束标点）
    tail = text[start:].strip()
    if len(tail) >= min_len:
        real_start = start + (len(text[start:]) - len(text[start:].lstrip()))
        sentences.append((tail, real_start))

    return sentences


# ==================================================================
# 三种分块策略
# ==================================================================
def chunk_by_sentence(doc, cfg) -> List[Chunk]:
    """按句切分。每个句子一个 Chunk。"""
    chunks = []
    sentences = split_sentences(doc.text, min_len=cfg.min_chunk_len)

    for i, (sent, pos) in enumerate(sentences):
        chunks.append(Chunk(
            text=sent,
            source=doc.source,
            chunk_id=i,
            char_start=pos,
            char_end=pos + len(sent),
            metadata={"strategy": "sentence"},
        ))

    logger.info(f"[Chunk] 按句切分: {len(chunks)} 块")
    return chunks


def chunk_by_paragraph(doc, cfg) -> List[Chunk]:
    """
    按段切分。段落由连续的非空行组成。

    注意：太长的段落要二次切分（按句），否则会超过 max_seq_len。
    """
    chunks = []
    paragraphs = re.split(r"\n\s*\n", doc.text)

    idx = 0
    cursor = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 找段落在原文中的位置
        pos = doc.text.find(para, cursor)
        if pos < 0:
            pos = cursor
        cursor = pos + len(para)

        if len(para) <= cfg.max_seq_len * 2:
            # 短段落整体作为一个 chunk
            if len(para) >= cfg.min_chunk_len:
                chunks.append(Chunk(
                    text=para,
                    source=doc.source,
                    chunk_id=idx,
                    char_start=pos,
                    char_end=pos + len(para),
                    metadata={"strategy": "paragraph"},
                ))
                idx += 1
        else:
            # 长段落二次切分
            sentences = split_sentences(para, min_len=cfg.min_chunk_len)
            for sent, sent_pos in sentences:
                chunks.append(Chunk(
                    text=sent,
                    source=doc.source,
                    chunk_id=idx,
                    char_start=pos + sent_pos,
                    char_end=pos + sent_pos + len(sent),
                    metadata={"strategy": "paragraph-sentence"},
                ))
                idx += 1

    logger.info(f"[Chunk] 按段切分: {len(chunks)} 块")
    return chunks


def chunk_by_sliding(doc, cfg) -> List[Chunk]:
    """
    滑窗切分：每 N 个相邻句子作为一个块，滑动 stride 个句子。

    好处：保留了句子间的上下文
    代价：块数变多（约 N/stride 倍），检索更慢
    """
    chunks = []
    sentences = split_sentences(doc.text, min_len=cfg.min_chunk_len)
    if not sentences:
        return chunks

    window = cfg.sliding_window
    stride = cfg.sliding_stride

    idx = 0
    for i in range(0, len(sentences), stride):
        window_sents = sentences[i:i + window]
        if not window_sents:
            break
        if len(window_sents) < window and i > 0:
            # 尾部不足一个窗口，也保留
            pass

        text = " ".join(s for s, _ in window_sents)
        start = window_sents[0][1]
        end = window_sents[-1][1] + len(window_sents[-1][0])

        chunks.append(Chunk(
            text=text,
            source=doc.source,
            chunk_id=idx,
            char_start=start,
            char_end=end,
            metadata={
                "strategy": "sliding",
                "sent_start": i,
                "sent_end": i + len(window_sents),
            },
        ))
        idx += 1

    logger.info(f"[Chunk] 滑窗切分 (window={window}, stride={stride}): {len(chunks)} 块")
    return chunks


# ==================================================================
# 统一入口
# ==================================================================
def chunk_document(doc, cfg) -> List[Chunk]:
    """根据配置选择分块策略。"""
    strategy = cfg.chunk_strategy
    if strategy == "sentence":
        return chunk_by_sentence(doc, cfg)
    elif strategy == "paragraph":
        return chunk_by_paragraph(doc, cfg)
    elif strategy == "sliding":
        return chunk_by_sliding(doc, cfg)
    else:
        raise ValueError(f"未知策略: {strategy}")


# ==================================================================
# 测试
# ==================================================================
if __name__ == "__main__":
    from parser import parse_document
    from config import CFG

    text = """自然语言处理是人工智能的重要分支。它研究如何让计算机理解和生成人类语言。
近年来，基于Transformer的模型在NLP任务上取得了巨大突破。

预训练语言模型如BERT、GPT等，通过在大规模语料上训练，学到了丰富的语言知识。
这些模型可以通过微调适应各种下游任务。

抄袭检测是NLP的一个重要应用。传统方法基于字符串匹配，但无法处理同义改写。
基于语义相似度的方法能更好地识别这类抄袭。"""

    Path = __import__("pathlib").Path
    Path("data/uploads/chunk_test.txt").write_text(text, encoding="utf-8")

    doc = parse_document("data/uploads/chunk_test.txt")
    print(f"\n文档: {doc.num_chars} 字符\n")

    for strategy in ["sentence", "paragraph", "sliding"]:
        CFG.chunk_strategy = strategy
        print(f"\n{'=' * 60}")
        print(f"策略: {strategy}")
        print(f"{'=' * 60}")
        chunks = chunk_document(doc, CFG)
        for c in chunks[:5]:
            print(f"  [{c.chunk_id}] pos={c.char_start}-{c.char_end}  {c.text[:50]}")
