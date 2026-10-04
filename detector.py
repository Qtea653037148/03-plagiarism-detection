"""
detector.py
抄袭检测核心 pipeline。

流程:
    待查文档 → 解析 → 分块 → 编码 → 与知识库检索 → 收集相似片段
             → 去重 → 分级 → 计算文档级抄袭比例 → 输出结构化结果
"""

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict
from collections import defaultdict

import numpy as np

from config import CFG
from parser import parse_document, Document
from chunker import chunk_document, Chunk
from knowledge_base import KnowledgeBase
from embedder import Embedder
from utils import setup_logger, normalize_for_match

logger = setup_logger()


# ==================================================================
# 数据类
# ==================================================================
@dataclass
class Match:
    """
    一条相似匹配：待查文档的某个块 与 知识库的某个块 相似。

    query_chunk:  待查文档中的块
    source_chunk: 知识库中命中的块
    score:        相似度（余弦）
    level:        分级（high / medium / low / none）
    """
    query_chunk: Chunk
    source_chunk: Chunk
    score: float
    level: str = "none"

    def to_dict(self):
        return {
            "query_text": self.query_chunk.text,
            "query_source": self.query_chunk.source,
            "query_pos": [self.query_chunk.char_start, self.query_chunk.char_end],
            "source_text": self.source_chunk.text,
            "source_file": self.source_chunk.source,
            "source_pos": [self.source_chunk.char_start, self.source_chunk.char_end],
            "score": round(self.score, 4),
            "level": self.level,
        }


@dataclass
class DetectionResult:
    """
    检测结果。

    query_doc:      待查文档
    matches:        所有相似片段（已去重、按分数排序）
    stats:          统计信息（文档级相似度、分级数量等）
    elapsed:        耗时（秒）
    """
    query_doc: Document
    matches: List[Match] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    elapsed: float = 0.0

    def high_matches(self) -> List[Match]:
        return [m for m in self.matches if m.level == "high"]

    def medium_matches(self) -> List[Match]:
        return [m for m in self.matches if m.level == "medium"]

    def is_plagiarism(self) -> bool:
        """根据文档级抄袭比例判定。"""
        return self.stats.get("plagiarism_ratio", 0) >= CFG.doc_plagiarism_ratio

    def to_dict(self):
        return {
            "query_doc": {
                "source": self.query_doc.source,
                "num_chars": self.query_doc.num_chars,
                "metadata": self.query_doc.metadata,
            },
            "matches": [m.to_dict() for m in self.matches],
            "stats": self.stats,
            "elapsed": self.elapsed,
            "is_plagiarism": self.is_plagiarism(),
        }


# ==================================================================
# 检测器
# ==================================================================
class PlagiarismDetector:
    """
    抄袭检测器。

    用法:
        kb = KnowledgeBase(CFG)
        kb.load()

        detector = PlagiarismDetector(CFG, kb)
        result = detector.detect("data/uploads/待查论文.txt")

        print(result.stats)
        for m in result.high_matches():
            print(m.score, m.query_chunk.text)
    """

    def __init__(self, cfg=CFG, knowledge_base: Optional[KnowledgeBase] = None):
        self.cfg = cfg
        # 复用知识库的 embedder，避免重复加载模型
        if knowledge_base is None:
            self.kb = KnowledgeBase(cfg)
        else:
            self.kb = knowledge_base
        self.embedder = self.kb.embedder

    # ----------------------------------------------------------------
    # 主流程
    # ----------------------------------------------------------------
    def detect(self, file_path: str) -> Optional[DetectionResult]:
        """
        检测文件是否抄袭知识库。

        返回 None 表示文件解析失败。
        """
        if self.kb.is_empty():
            logger.error("[Detector] 知识库为空，请先 build")
            return None

        # 1. 解析
        doc = parse_document(file_path)
        if doc is None:
            logger.error(f"[Detector] 解析失败: {file_path}")
            return None
        return self._detect_document(doc)

    def detect_text(self, text: str, source_name: str = "<text>") -> DetectionResult:
        """直接检测一段文本（不经过文件）。"""
        from parser import Document as Doc
        from utils import text_hash
        doc = Doc(
            source=source_name,
            text=text,
            metadata={"format": "raw_text"},
            file_hash=text_hash(text),
        )
        return self._detect_document(doc)

    def _detect_document(self, doc: Document) -> DetectionResult:
        """核心逻辑。"""
        t0 = time.time()
        logger.info(f"[Detector] 检测文档: {Path(doc.source).name} "
                    f"({doc.num_chars} 字符)")

        # 2. 分块
        chunks = chunk_document(doc, self.cfg)
        if not chunks:
            logger.warning("[Detector] 文档没有可检测的块")
            return DetectionResult(
                query_doc=doc, stats={"error": "empty_document"},
                elapsed=time.time() - t0,
            )

        # 3. 批量检索
        logger.info(f"[Detector] 检索 {len(chunks)} 个块...")
        queries = [c.text for c in chunks]
        batch_results = self.kb.search_batch(queries, top_k=self.cfg.recall_top_k)

        # 4. 收集匹配
        matches = self._collect_matches(chunks, batch_results)

        # 5. 去重（同一对 (query, source) 只保留最高分）
        matches = self._deduplicate(matches)

        # 6. 过滤低分
        matches = [m for m in matches if m.score >= self.cfg.threshold_low]

        # 7. 分级
        for m in matches:
            m.level = self._classify(m.score)

        # 8. 按分数排序
        matches.sort(key=lambda x: -x.score)

        # 9. 统计
        stats = self._compute_stats(chunks, matches, doc)

        elapsed = time.time() - t0
        logger.info(
            f"[Detector] 完成，耗时 {elapsed:.2f}s，"
            f"匹配 {len(matches)} 条 "
            f"(high={stats['high_count']}, "
            f"medium={stats['medium_count']}, "
            f"low={stats['low_count']})"
        )

        return DetectionResult(
            query_doc=doc,
            matches=matches,
            stats=stats,
            elapsed=elapsed,
        )

    # ----------------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------------
    def _collect_matches(
        self, chunks: List[Chunk], batch_results: List[List[dict]]
    ) -> List[Match]:
        """把检索结果转换为 Match 列表。"""
        matches = []
        for chunk, results in zip(chunks, batch_results):
            for r in results:
                matches.append(Match(
                    query_chunk=chunk,
                    source_chunk=r["chunk"],
                    score=r["score"],
                ))
        return matches

    def _deduplicate(self, matches: List[Match]) -> List[Match]:
        """
        去重：同一对 (query_chunk.hash, source_chunk.hash) 只保留最高分。
        同一段文本在知识库里可能出现多次（重复文档），这里合并。
        """
        best = {}
        for m in matches:
            key = (m.query_chunk.hash, m.source_chunk.hash)
            if key not in best or m.score > best[key].score:
                best[key] = m
        return list(best.values())

    def _classify(self, score: float) -> str:
        """按分数分级。"""
        if score >= self.cfg.threshold_high:
            return "high"
        elif score >= self.cfg.threshold_medium:
            return "medium"
        elif score >= self.cfg.threshold_low:
            return "low"
        return "none"

    def _compute_stats(
        self, chunks: List[Chunk], matches: List[Match], doc: Document
    ) -> dict:
        """
        计算文档级统计。

        关键指标:
            plagiarism_ratio: 被判定为 high 的块占全部块的比例
            char_ratio:       被判定为 high 的块的字数占全文的比例
            source_distribution: 抄袭来源的分布（哪个文档被抄了多少）
        """
        total_chunks = len(chunks)
        high_count = sum(1 for m in matches if m.level == "high")
        medium_count = sum(1 for m in matches if m.level == "medium")
        low_count = sum(1 for m in matches if m.level == "low")

        # 被 high 命中的 query chunk 集合（去重）
        high_query_hashes = {m.query_chunk.hash for m in matches
                             if m.level == "high"}
        medium_query_hashes = {m.query_chunk.hash for m in matches
                               if m.level == "medium"}

        # 字符级比例
        total_chars = sum(len(c.text) for c in chunks)
        high_chars = sum(len(c.text) for c in chunks
                         if c.hash in high_query_hashes)
        medium_chars = sum(len(c.text) for c in chunks
                           if c.hash in medium_query_hashes)

        # 来源分布
        source_dist = defaultdict(lambda: {"count": 0, "chars": 0, "max_score": 0.0})
        for m in matches:
            if m.level in ("high", "medium"):
                key = Path(m.source_chunk.source).name
                source_dist[key]["count"] += 1
                source_dist[key]["chars"] += len(m.query_chunk.text)
                source_dist[key]["max_score"] = max(
                    source_dist[key]["max_score"], m.score
                )

        plagiarism_ratio = high_count / max(1, total_chunks)
        char_ratio = high_chars / max(1, total_chars)

        return {
            "total_chunks": total_chunks,
            "total_chars": total_chars,
            "high_count": high_count,
            "medium_count": medium_count,
            "low_count": low_count,
            "high_ratio": round(high_count / max(1, total_chunks), 4),
            "plagiarism_ratio": round(plagiarism_ratio, 4),
            "char_ratio": round(char_ratio, 4),
            "medium_char_ratio": round(medium_chars / max(1, total_chars), 4),
            "source_distribution": dict(source_dist),
            "is_plagiarism": plagiarism_ratio >= self.cfg.doc_plagiarism_ratio,
        }

    # ----------------------------------------------------------------
    # 打印摘要
    # ----------------------------------------------------------------
    def print_summary(self, result: DetectionResult, max_show: int = 20):
        """控制台打印检测报告。"""
        print("\n" + "=" * 70)
        print(f"抄袭检测报告: {Path(result.query_doc.source).name}")
        print("=" * 70)

        s = result.stats
        print(f"\n【文档信息】")
        print(f"  字符数:     {s.get('total_chars', 0)}")
        print(f"  分块数:     {s.get('total_chunks', 0)}")
        print(f"  检测耗时:   {result.elapsed:.2f}s")

        print(f"\n【匹配统计】")
        print(f"  高度相似 (score≥{self.cfg.threshold_high}):  {s.get('high_count', 0)} 条")
        print(f"  中度相似 (score≥{self.cfg.threshold_medium}):  {s.get('medium_count', 0)} 条")
        print(f"  低度相似 (score≥{self.cfg.threshold_low}):  {s.get('low_count', 0)} 条")

        print(f"\n【抄袭判定】")
        print(f"  高度相似块占比:  {s.get('high_ratio', 0):.2%}")
        print(f"  高度相似字符占比: {s.get('char_ratio', 0):.2%}")
        print(f"  判定阈值:        {self.cfg.doc_plagiarism_ratio:.2%}")
        verdict = "⚠️ 判定为抄袭" if result.is_plagiarism() else "✅ 未判定为抄袭"
        print(f"  结论:            {verdict}")

        if s.get("source_distribution"):
            print(f"\n【相似来源分布】")
            for src, info in s["source_distribution"].items():
                print(f"  {src}")
                print(f"    命中 {info['count']} 条, "
                      f"字数 {info['chars']}, "
                      f"最高相似度 {info['max_score']:.4f}")

        if result.matches:
            print(f"\n【Top 相似片段】")
            for i, m in enumerate(result.matches[:max_show], 1):
                q_text = m.query_chunk.text[:50].replace("\n", " ")
                s_text = m.source_chunk.text[:50].replace("\n", " ")
                src_name = Path(m.source_chunk.source).name
                print(f"\n  [{i}] [{m.level.upper()}] score={m.score:.4f}")
                print(f"      待查: {q_text}...")
                print(f"      来源: {s_text}...  ({src_name})")

        print("\n" + "=" * 70)


# ==================================================================
# 测试
# ==================================================================
if __name__ == "__main__":
    from pathlib import Path
    logger.info("=" * 60)
    logger.info("准备测试数据")
    logger.info("=" * 60)

    # 准备一个待查文档：混合抄袭和原创
    test_doc = """自然语言处理是人工智能的重要分支。它研究如何让计算机理解和生成人类语言。
近年来，基于Transformer的模型在NLP任务上取得了巨大突破。
北京是中国的首都，有着悠久的历史和丰富的文化资源。
我今天下午去咖啡馆看了一本书，关于咖啡的历史。
办理信用卡可以通过银行网点、官网或手机APP进行。
这段文字是原创的，讨论一些不存在的话题比如火星移民计划。
信用卡申请需要满足一定条件。
"""

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / "suspected.txt").write_text(test_doc, encoding="utf-8")

    # 加载知识库
    logger.info("")
    logger.info("=" * 60)
    logger.info("加载知识库")
    logger.info("=" * 60)
    kb = KnowledgeBase(CFG)
    if not kb.load():
        logger.error("知识库为空，请先运行 knowledge_base.py 建库")
        exit(1)

    # 检测
    detector = PlagiarismDetector(CFG, kb)
    result = detector.detect(str(upload_dir / "suspected.txt"))

    if result:
        detector.print_summary(result)
