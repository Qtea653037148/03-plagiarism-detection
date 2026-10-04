"""
web_search.py
网络原文定位：输入一段文本，联网搜索找出可能的原文来源。

后端: cn.bing.com（必应中国）
    - 国内可直连
    - 反爬比百度松
    - 结果结构清晰：li.b_algo 是每个结果容器

流程:
    1. 分句，取最长的 N 句作为查询
    2. 每句发到 cn.bing.com，解析 HTML
    3. 提取标题、链接、摘要
    4. 用 BiEncoder 计算摘要与原句的相似度
    5. 排序返回
"""

import time
import re
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from config import CFG
from embedder import Embedder
from utils import setup_logger, truncate

logger = setup_logger()


# ==================================================================
# 数据类
# ==================================================================
@dataclass
class WebMatch:
    """一条网络匹配结果。"""
    query: str          # 原查询句
    title: str          # 搜索结果标题
    body: str           # 搜索结果摘要
    url: str            # 链接
    score: float        # 与原句的相似度
    search_query: str = ""   # 实际使用的搜索词

    def to_dict(self):
        return {
            "query": self.query,
            "title": self.title,
            "body": self.body,
            "url": self.url,
            "score": round(self.score, 4),
            "search_query": self.search_query,
        }


# ==================================================================
# 网络搜索器
# ==================================================================
class WebSearcher:
    """基于 cn.bing.com 的联网原文定位。"""

    BASE_URL = "https://cn.bing.com/search"

    # 浏览器 UA，避免被识别为爬虫
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

    def __init__(self, embedder: Optional[Embedder] = None, cfg=CFG):
        self.cfg = cfg
        self.embedder = embedder or Embedder(cfg, verbose=False)
        # 复用 session 以减少握手开销
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    # ----------------------------------------------------------------
    # HTML 解析
    # ----------------------------------------------------------------
    def _parse_bing_results(self, html: str, max_results: int = 10) -> List[dict]:
        """
        解析必应搜索结果页。

        每个结果的结构（2024-2026 年验证）：
            <li class="b_algo">
                <h2><a href="...">标题</a></h2>
                <div class="b_caption">
                    <p>摘要</p>
                </div>
            </li>
        """
        soup = BeautifulSoup(html, "lxml")
        items = soup.select("li.b_algo")
        results = []

        for item in items[:max_results]:
            # 标题和链接
            title_el = item.select_one("h2 a")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            url = title_el.get("href", "")

            # 摘要：尝试多个选择器
            body = ""
            for sel in ["div.b_caption p", ".b_caption", "p", ".b_lineclamp4"]:
                body_el = item.select_one(sel)
                if body_el:
                    body = body_el.get_text(strip=True)
                    break

            if title and url:
                results.append({"title": title, "body": body, "url": url})

        return results

    # ----------------------------------------------------------------
    # 搜索引擎请求
    # ----------------------------------------------------------------
    def _fetch_bing(self, query: str, max_results: int = 10) -> List[dict]:
        """发一个搜索请求，返回解析后的结果列表。"""
        params = {
            "q": query,
            "count": max_results,
            "setlang": "zh-CN",
            "ensearch": "0",
        }

        try:
            resp = self.session.get(
                self.BASE_URL,
                params=params,
                timeout=15,
                allow_redirects=True,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"[WebSearch] 请求失败: {e}")
            return []

        if resp.status_code != 200:
            logger.warning(f"[WebSearch] 状态码 {resp.status_code}")
            return []

        return self._parse_bing_results(resp.text, max_results=max_results)

    # ----------------------------------------------------------------
    # 关键词提取
    # ----------------------------------------------------------------
    @staticmethod
    def extract_keywords(sentence: str, max_words: int = 8) -> str:
        """从句子中抽取关键词，用于搜索的第二路查询。"""
        try:
            import jieba.analyse
            keywords = jieba.analyse.extract_tags(sentence, topK=max_words)
            return " ".join(keywords)
        except Exception:
            words = re.findall(r"[\u4e00-\u9fa5]{2,}", sentence)
            return " ".join(words[:max_words])

    # ----------------------------------------------------------------
    # 单句搜索
    # ----------------------------------------------------------------
    def search_sentence(self, sentence: str, max_results: int = 10) -> List[WebMatch]:
        """
        用一句话联网搜索，返回按相似度排序的结果。
        """
        raw_results = []

        # ---------- 查询 1: 整句（截断到 50 字，太长搜不到东西） ----------
        full_query = sentence.strip()[:50]
        if full_query:
            logger.info(f"[WebSearch] 查询(整句): {truncate(full_query, 40)}")
            results = self._fetch_bing(full_query, max_results=max_results)
            for r in results:
                r["_query_type"] = "full"
                r["_search_query"] = full_query
            raw_results.extend(results)
            time.sleep(1.0)   # 反爬

        # ---------- 查询 2: 关键词 ----------
        kw_query = self.extract_keywords(sentence)
        if kw_query and kw_query != full_query:
            logger.info(f"[WebSearch] 查询(关键词): {truncate(kw_query, 40)}")
            results = self._fetch_bing(kw_query, max_results=max_results)
            for r in results:
                r["_query_type"] = "keyword"
                r["_search_query"] = kw_query
            raw_results.extend(results)

        if not raw_results:
            return []

        # ---------- 去重 ----------
        seen_urls = set()
        unique = []
        for r in raw_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique.append(r)

        logger.info(f"[WebSearch] 去重后 {len(unique)} 条结果")

        # ---------- 相似度计算 ----------
        texts = [sentence] + [
            f"{r['title']}。{r['body']}" for r in unique
        ]
        embs = self.embedder.encode(texts)
        sims = (embs[0:1] @ embs[1:].T)[0]

        matches = []
        for r, s in zip(unique, sims):
            matches.append(WebMatch(
                query=sentence,
                title=r.get("title", ""),
                body=r.get("body", ""),
                url=r.get("url", ""),
                score=float(s),
                search_query=r.get("_search_query", ""),
            ))

        matches.sort(key=lambda m: -m.score)
        return matches

    # ----------------------------------------------------------------
    # 多句搜索
    # ----------------------------------------------------------------
    def search_text(
        self,
        text: str,
        top_k_sentences: int = 3,
        max_results: int = 8,
    ) -> List[WebMatch]:
        """
        对一段文本搜索。
        取最长的 N 句（长句信息量更大）。
        """
        from chunker import split_sentences

        sentences = split_sentences(text, min_len=10)
        if not sentences:
            return []

        # 按长度降序取前 N 句
        sorted_sents = sorted(sentences, key=lambda x: -len(x[0]))
        selected = sorted_sents[:top_k_sentences]

        logger.info(f"[WebSearch] 对 {len(selected)} 个句子联网搜索")

        all_matches = []
        for sent, _ in selected:
            results = self.search_sentence(sent, max_results=max_results)
            # 每句保留 top-3
            all_matches.extend(results[:3])

        all_matches.sort(key=lambda m: -m.score)
        return all_matches

    # ----------------------------------------------------------------
    # 对检测结果的 high 片段搜索
    # ----------------------------------------------------------------
    def search_detection_result(
        self, detection_result, max_sentences: int = 3
    ) -> List[WebMatch]:
        """
        对 PlagiarismDetector 的结果中 high 级别的片段做网络搜索。
        """
        high_matches = detection_result.high_matches()
        if not high_matches:
            logger.info("[WebSearch] 没有 high 级别片段")
            return []

        selected = high_matches[:max_sentences]
        all_matches = []
        for m in selected:
            results = self.search_sentence(
                m.query_chunk.text,
                max_results=self.cfg.web_search_max_results,
            )
            all_matches.extend(results[:3])

        all_matches.sort(key=lambda x: -x.score)
        return all_matches


# ==================================================================
# 打印
# ==================================================================
def print_web_matches(matches: List[WebMatch], max_show: int = 10):
    print("\n" + "=" * 70)
    print(f"网络搜索匹配结果（共 {len(matches)} 条）")
    print("=" * 70)
    for i, m in enumerate(matches[:max_show], 1):
        print(f"\n[{i}] score={m.score:.4f}")
        print(f"    原句: {truncate(m.query, 60)}")
        print(f"    标题: {m.title[:80]}")
        print(f"    摘要: {m.body[:120]}")
        print(f"    链接: {m.url}")


# ==================================================================
# 测试
# ==================================================================
if __name__ == "__main__":
    import sys

    searcher = WebSearcher(cfg=CFG)

    if len(sys.argv) > 1:
        # 命令行模式：python web_search.py "要搜索的句子"
        query = " ".join(sys.argv[1:])
        logger.info(f"搜索: {query}")
        matches = searcher.search_sentence(query, max_results=8)
        print_web_matches(matches, max_show=8)
    else:
        # 测试模式
        logger.info("=" * 60)
        logger.info("WebSearcher 单句搜索测试")
        logger.info("=" * 60)

        test_sentences = [
            "自然语言处理是人工智能的重要分支",
            "北京是中国的首都，有着悠久的历史和丰富的文化资源",
        ]

        for sent in test_sentences:
            logger.info(f"\n搜索: {sent}")
            matches = searcher.search_sentence(sent, max_results=5)
            if matches:
                for m in matches[:3]:
                    logger.info(f"  {m.score:.4f}  {m.title[:50]}")
                    logger.info(f"         {m.url}")
            else:
                logger.info("  没有结果")
            time.sleep(2)

        logger.info("")
        logger.info("=" * 60)
        logger.info("多句搜索测试")
        logger.info("=" * 60)

        text = """自然语言处理是人工智能的重要分支。它研究如何让计算机理解和生成人类语言。
近年来，基于Transformer的模型在NLP任务上取得了巨大突破。"""

        matches = searcher.search_text(text, top_k_sentences=2, max_results=5)
        print_web_matches(matches, max_show=10)
