"""
knowledge_base.py
知识库：管理一批文档，提供检索能力。

功能:
    - 从目录批量导入文档
    - 解析 → 分块 → 编码 → 建 FAISS 索引
    - 增量添加文档
    - 保存/加载索引
    - 检索 Top-K 相似块

存储布局:
    data/index/
        ├── index.faiss      FAISS 索引
        └── metadata.pkl     块列表 + 文档信息
"""

import os
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from config import CFG
from parser import parse_document, parse_directory, Document
from chunker import chunk_document, Chunk
from embedder import Embedder
from utils import setup_logger, file_hash

logger = setup_logger()


class KnowledgeBase:
    """
    知识库。

    用法:
        kb = KnowledgeBase(CFG)
        kb.build_from_directory("data/knowledge_base")   # 建库
        kb.save()                                        # 保存
        results = kb.search("查询文本", top_k=5)         # 检索

        # 或者加载已有的库
        kb = KnowledgeBase(CFG)
        kb.load()
        results = kb.search("查询", top_k=5)
    """

    def __init__(self, cfg=CFG, embedder: Optional[Embedder] = None):
        self.cfg = cfg
        # 允许外部注入 embedder（避免重复加载模型）
        self.embedder = embedder if embedder is not None else Embedder(cfg)

        self.chunks: List[Chunk] = []
        self.documents: List[Document] = []
        self.index: Optional[faiss.Index] = None

        # 记录已导入文件的 hash，用于去重
        self._seen_hashes: set = set()

    # ================================================================
    # 构建
    # ================================================================
    def build_from_directory(self, dir_path: str, recursive: bool = True):
        """从目录批量导入文档，重建整个知识库。"""
        logger.info(f"[KB] 从目录导入: {dir_path}")
        docs = parse_directory(dir_path, recursive=recursive)

        if not docs:
            logger.warning("[KB] 目录中没有可用文档")
            return

        self.documents = []
        self.chunks = []
        self._seen_hashes = set()

        for doc in docs:
            self._add_document_to_memory(doc)

        self._build_index()

    def add_document(self, file_path: str) -> bool:
        """
        增量添加单个文档。
        返回是否成功添加（重复或解析失败返回 False）。
        """
        doc = parse_document(file_path)
        if doc is None:
            return False

        if doc.file_hash in self._seen_hashes:
            logger.info(f"[KB] 跳过重复文档: {Path(file_path).name}")
            return False

        self._add_document_to_memory(doc)

        # 增量编码和索引
        new_chunks = [c for c in self.chunks
                      if c.source == doc.source]
        if not new_chunks:
            return False

        vecs = self.embedder.encode([c.text for c in new_chunks])
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

        logger.info(f"[KB] 增量添加: {Path(file_path).name}, "
                    f"{len(new_chunks)} 块")
        return True

    def _add_document_to_memory(self, doc: Document):
        """把文档分块，追加到 chunks 列表。"""
        chunks = chunk_document(doc, self.cfg)
        self.documents.append(doc)
        self.chunks.extend(chunks)
        self._seen_hashes.add(doc.file_hash)
        logger.info(f"[KB] 导入: {Path(doc.source).name}, {len(chunks)} 块")

    def _build_index(self):
        """对所有 chunks 编码并建 FAISS 索引。"""
        if not self.chunks:
            logger.warning("[KB] 没有块可编码")
            return

        logger.info(f"[KB] 编码 {len(self.chunks)} 个块...")
        t0 = time.time()
        vecs = self.embedder.encode(
            [c.text for c in self.chunks], show_progress=True
        )
        encode_time = time.time() - t0
        logger.info(f"[KB] 编码完成，耗时 {encode_time:.1f}s")

        # 建索引
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        logger.info(f"[KB] 索引建立完成: {self.index.ntotal} 个向量, "
                    f"维度 {self.index.d}")

    # ================================================================
    # 检索
    # ================================================================
    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """
        检索与 query 最相似的 top_k 个块。

        返回: [{"chunk": Chunk, "score": float, "rank": int}, ...]
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("[KB] 索引为空")
            return []

        q_vec = self.embedder.encode_one(query)
        scores, indices = self.index.search(q_vec.reshape(1, -1), top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            results.append({
                "chunk": self.chunks[idx],
                "score": float(score),
                "rank": rank,
            })
        return results

    def search_batch(self, queries: List[str], top_k: int = 10) -> List[List[dict]]:
        """批量检索，比逐个调用快。"""
        if self.index is None or self.index.ntotal == 0:
            return [[] for _ in queries]

        q_vecs = self.embedder.encode(queries)
        scores, indices = self.index.search(q_vecs, top_k)

        all_results = []
        for i in range(len(queries)):
            results = []
            for rank, (score, idx) in enumerate(zip(scores[i], indices[i])):
                if idx < 0:
                    continue
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(score),
                    "rank": rank,
                })
            all_results.append(results)
        return all_results

    # ================================================================
    # 持久化
    # ================================================================
    def save(self, save_dir: Optional[str] = None):
        """保存索引和元数据到磁盘。"""
        save_dir = save_dir or self.cfg.index_dir
        os.makedirs(save_dir, exist_ok=True)

        if self.index is None:
            logger.warning("[KB] 索引为空，跳过保存")
            return

        # 保存 FAISS 索引
        index_path = os.path.join(save_dir, "index.faiss")
        faiss.write_index(self.index, index_path)
        logger.info(f"[KB] 索引保存: {index_path}")

        # 保存 chunks 和文档元数据
        meta_path = os.path.join(save_dir, "metadata.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "documents": self.documents,
                "config": {
                    "model_name": self.cfg.biencoder_model_name,
                    "pooling": self.cfg.biencoder_pooling,
                    "max_seq_len": self.cfg.max_seq_len,
                    "chunk_strategy": self.cfg.chunk_strategy,
                },
            }, f)
        logger.info(f"[KB] 元数据保存: {meta_path}")

        # 统计
        size_mb = os.path.getsize(index_path) / 1e6
        logger.info(f"[KB] 索引文件: {size_mb:.2f} MB, "
                    f"{len(self.chunks)} 块, "
                    f"{len(self.documents)} 个文档")

    def load(self, save_dir: Optional[str] = None) -> bool:
        """从磁盘加载。返回是否成功。"""
        save_dir = save_dir or self.cfg.index_dir

        index_path = os.path.join(save_dir, "index.faiss")
        meta_path = os.path.join(save_dir, "metadata.pkl")

        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            logger.warning(f"[KB] 找不到索引文件，请先 build")
            return False

        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.chunks = meta["chunks"]
        self.documents = meta["documents"]
        self._seen_hashes = {d.file_hash for d in self.documents}

        logger.info(f"[KB] 加载索引: {self.index.ntotal} 个向量, "
                    f"{len(self.chunks)} 块, "
                    f"{len(self.documents)} 个文档")
        return True

    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    def stats(self) -> dict:
        return {
            "num_documents": len(self.documents),
            "num_chunks": len(self.chunks),
            "num_vectors": self.index.ntotal if self.index else 0,
            "index_dim": self.index.d if self.index else 0,
        }


# ==================================================================
# 测试
# ==================================================================
if __name__ == "__main__":
    # 准备测试文档
    kb_dir = Path("data/knowledge_base")
    kb_dir.mkdir(parents=True, exist_ok=True)

    doc1 = """自然语言处理是人工智能的重要分支。它研究如何让计算机理解和生成人类语言。
近年来，基于Transformer的模型在NLP任务上取得了巨大突破。
预训练语言模型如BERT、GPT等，通过在大规模语料上训练，学到了丰富的语言知识。"""

    doc2 = """信用卡申请需要满足一定条件。通常要求申请人有稳定的收入来源和良好的信用记录。
办理信用卡可以通过银行网点、官网或手机APP进行。
申请时需要提供身份证明、收入证明等材料。"""

    doc3 = """北京是中国的首都，有着悠久的历史和丰富的文化资源。
故宫、天安门、长城都是著名的旅游景点。
每年吸引大量国内外游客前来参观。"""

    (kb_dir / "nlp_intro.txt").write_text(doc1, encoding="utf-8")
    (kb_dir / "credit_card.txt").write_text(doc2, encoding="utf-8")
    (kb_dir / "beijing.txt").write_text(doc3, encoding="utf-8")

    # 建库
    logger.info("=" * 60)
    logger.info("构建知识库")
    logger.info("=" * 60)
    kb = KnowledgeBase(CFG)
    kb.build_from_directory(str(kb_dir))

    logger.info("")
    logger.info(f"统计: {kb.stats()}")

    # 保存
    kb.save()

    # 检索测试
    logger.info("")
    logger.info("=" * 60)
    logger.info("检索测试")
    logger.info("=" * 60)

    queries = [
        "如何申请信用卡",
        "北京有什么好玩的地方",
        "Transformer模型在NLP中的应用",
        "今天午饭吃什么",            # 无关查询
    ]

    for q in queries:
        logger.info(f"\n查询: {q}")
        results = kb.search(q, top_k=3)
        for r in results:
            c = r["chunk"]
            logger.info(
                f"  [{r['rank']}] {r['score']:.4f}  "
                f"{Path(c.source).name}  {c.text[:50]}"
            )
