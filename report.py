"""
report.py
生成抄袭检测的可视化 HTML 报告。

报告内容:
    1. 概览卡片（文档信息、抄袭比例、判定结论）
    2. 相似来源分布（柱状图）
    3. 相似片段列表（带高亮）
    4. 网络搜索结果（如果有）
    5. 原文高亮视图（把抄袭片段在原文档中标出）
"""

import html
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from config import CFG
from utils import setup_logger

if TYPE_CHECKING:
    from detector import DetectionResult
    from web_search import WebMatch

logger = setup_logger()


# ==================================================================
# HTML 转义
# ==================================================================
def esc(text: str) -> str:
    """HTML 转义 + 换行转 <br>。"""
    return html.escape(text).replace("\n", "<br>")


# ==================================================================
# 高亮原文
# ==================================================================
def highlight_document(doc_text: str, matches: List, levels=("high", "medium")) -> str:
    """
    在原文中高亮被判定为抄袭的片段。

    做法:
        收集所有 high/medium 的 query_chunk 的位置区间
        合并重叠区间
        按位置切分原文，给这些区间加 <mark> 标签
    """
    # 收集区间
    intervals = []
    for m in matches:
        if m.level in levels:
            c = m.query_chunk
            intervals.append((c.char_start, c.char_end, m.level))

    if not intervals:
        return esc(doc_text)

    # 按起始位置排序并合并重叠
    intervals.sort()
    merged = []
    for start, end, level in intervals:
        if merged and start <= merged[-1][1]:
            # 重叠，扩展结束位置，level 取更高的
            prev_start, prev_end, prev_level = merged[-1]
            new_level = "high" if "high" in (prev_level, level) else prev_level
            merged[-1] = (prev_start, max(prev_end, end), new_level)
        else:
            merged.append((start, end, level))

    # 拼接 HTML
    parts = []
    cursor = 0
    for start, end, level in merged:
        if start > cursor:
            parts.append(esc(doc_text[cursor:start]))
        # 高亮样式
        cls = "high" if level == "high" else "medium"
        parts.append(
            f'<mark class="hl-{cls}">{esc(doc_text[start:end])}</mark>'
        )
        cursor = end
    if cursor < len(doc_text):
        parts.append(esc(doc_text[cursor:]))

    return "".join(parts)


# ==================================================================
# HTML 模板
# ==================================================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>抄袭检测报告 - {doc_name}</title>
<style>
    * {{ box-sizing: border-box; }}
    body {{
        font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
        max-width: 1100px;
        margin: 0 auto;
        padding: 30px 20px;
        background: #f5f7fa;
        color: #2c3e50;
        line-height: 1.6;
    }}
    h1 {{ font-size: 26px; margin-bottom: 8px; }}
    h2 {{ font-size: 20px; margin-top: 36px; padding-bottom: 8px;
         border-bottom: 2px solid #e1e8ed; }}
    h3 {{ font-size: 16px; margin-top: 20px; }}
    .subtitle {{ color: #7f8c8d; font-size: 14px; margin-bottom: 24px; }}

    .card {{
        background: white;
        border-radius: 10px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }}

    .verdict {{
        padding: 16px 24px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 24px;
    }}
    .verdict.plagiarism {{
        background: #ffebee;
        color: #c62828;
        border-left: 6px solid #c62828;
    }}
    .verdict.clean {{
        background: #e8f5e9;
        color: #2e7d32;
        border-left: 6px solid #2e7d32;
    }}

    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 16px;
        margin-bottom: 20px;
    }}
    .stat {{
        background: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }}
    .stat .value {{
        font-size: 26px;
        font-weight: bold;
        color: #2c3e50;
    }}
    .stat .label {{
        font-size: 13px;
        color: #7f8c8d;
        margin-top: 4px;
    }}

    .match-item {{
        border-left: 4px solid #ccc;
        padding: 16px 20px;
        margin-bottom: 14px;
        background: #fafbfc;
        border-radius: 0 6px 6px 0;
    }}
    .match-item.high {{ border-color: #e74c3c; background: #fff5f5; }}
    .match-item.medium {{ border-color: #f39c12; background: #fffbf0; }}
    .match-item.low {{ border-color: #95a5a6; }}

    .match-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
        font-size: 13px;
        color: #7f8c8d;
    }}
    .badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }}
    .badge.high {{ background: #e74c3c; color: white; }}
    .badge.medium {{ background: #f39c12; color: white; }}
    .badge.low {{ background: #95a5a6; color: white; }}

    .pair {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-top: 8px;
    }}
    .pair .side {{
        background: white;
        padding: 12px;
        border-radius: 6px;
        font-size: 14px;
    }}
    .pair .side .side-label {{
        font-size: 12px;
        color: #7f8c8d;
        margin-bottom: 6px;
        text-transform: uppercase;
    }}

    mark.hl-high {{
        background: #ffcdd2;
        border-bottom: 2px solid #e74c3c;
        padding: 1px 2px;
    }}
    mark.hl-medium {{
        background: #ffe0b2;
        border-bottom: 2px solid #f39c12;
        padding: 1px 2px;
    }}

    .doc-view {{
        background: white;
        padding: 24px;
        border-radius: 10px;
        font-size: 15px;
        line-height: 2;
        white-space: pre-wrap;
        word-wrap: break-word;
    }}

    .web-item {{
        padding: 14px 18px;
        border-left: 3px solid #3498db;
        background: #f0f7ff;
        border-radius: 0 6px 6px 0;
        margin-bottom: 12px;
    }}
    .web-item a {{
        color: #2980b9;
        text-decoration: none;
        font-weight: 500;
    }}
    .web-item a:hover {{ text-decoration: underline; }}
    .web-item .url {{ font-size: 12px; color: #7f8c8d; }}

    .source-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 16px;
        background: #f8fafc;
        border-radius: 6px;
        margin-bottom: 8px;
    }}
    .source-bar .name {{ font-weight: 500; }}
    .source-bar .info {{ color: #7f8c8d; font-size: 13px; }}

    footer {{
        text-align: center;
        color: #95a5a6;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #e1e8ed;
    }}
</style>
</head>
<body>

<h1>🔍 抄袭检测报告</h1>
<div class="subtitle">
    文档: <strong>{doc_name}</strong> ·
    生成时间: {timestamp}
</div>

{verdict_html}

<div class="stats-grid">
    <div class="stat">
        <div class="value">{num_chars}</div>
        <div class="label">文档字数</div>
    </div>
    <div class="stat">
        <div class="value">{num_chunks}</div>
        <div class="label">文本块数</div>
    </div>
    <div class="stat">
        <div class="value">{high_count}</div>
        <div class="label">高度相似</div>
    </div>
    <div class="stat">
        <div class="value">{medium_count}</div>
        <div class="label">中度相似</div>
    </div>
    <div class="stat">
        <div class="value">{char_ratio:.1%}</div>
        <div class="label">抄袭字符占比</div>
    </div>
    <div class="stat">
        <div class="value">{elapsed:.2f}s</div>
        <div class="label">检测耗时</div>
    </div>
</div>

{source_section}

{match_section}

{web_section}

<h2>📄 原文高亮</h2>
<div class="card">
    <div class="doc-view">{highlighted_text}</div>
</div>

<footer>
    由抄袭检测系统生成 · 模型: {model_name} · 检测块数: {num_chunks}
</footer>

</body>
</html>
"""


# ==================================================================
# 报告生成器
# ==================================================================
class ReportGenerator:
    def __init__(self, cfg=CFG):
        self.cfg = cfg

    def generate(
        self,
        result: "DetectionResult",
        web_matches: Optional[List] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        生成 HTML 报告。

        返回报告文件路径。
        """
        output_path = output_path or os.path.join(
            self.cfg.figures_dir, "report.html"
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        s = result.stats
        doc_name = Path(result.query_doc.source).name

        # ---------- verdict ----------
        is_plag = result.is_plagiarism()
        verdict_html = (
            '<div class="verdict plagiarism">⚠️ 判定为抄袭</div>'
            if is_plag else
            '<div class="verdict clean">✅ 未判定为抄袭</div>'
        )

        # ---------- source section ----------
        source_section = self._build_source_section(s)

        # ---------- match section ----------
        match_section = self._build_match_section(result)

        # ---------- web section ----------
        web_section = self._build_web_section(web_matches)

        # ---------- highlighted text ----------
        highlighted = highlight_document(
            result.query_doc.text, result.matches
        )

        html_content = HTML_TEMPLATE.format(
            doc_name=esc(doc_name),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            verdict_html=verdict_html,
            num_chars=s.get("total_chars", 0),
            num_chunks=s.get("total_chunks", 0),
            high_count=s.get("high_count", 0),
            medium_count=s.get("medium_count", 0),
            char_ratio=s.get("char_ratio", 0.0),
            elapsed=result.elapsed,
            source_section=source_section,
            match_section=match_section,
            web_section=web_section,
            highlighted_text=highlighted,
            model_name=self.cfg.biencoder_model_name,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"[Report] 报告已生成: {output_path}")
        return output_path

    # ----------------------------------------------------------------
    def _build_source_section(self, stats: dict) -> str:
        dist = stats.get("source_distribution", {})
        if not dist:
            return ""

        rows = []
        for name, info in sorted(
            dist.items(), key=lambda x: -x[1]["count"]
        ):
            rows.append(f"""
            <div class="source-bar">
                <span class="name">📄 {esc(name)}</span>
                <span class="info">
                    命中 {info['count']} 条 ·
                    {info['chars']} 字 ·
                    最高相似度 {info['max_score']:.4f}
                </span>
            </div>""")

        return f"""
        <h2>📚 相似来源分布</h2>
        <div class="card">
            {"".join(rows)}
        </div>"""

    # ----------------------------------------------------------------
    def _build_match_section(self, result: "DetectionResult") -> str:
        if not result.matches:
            return """
            <h2>🔍 相似片段</h2>
            <div class="card">没有检测到相似片段。</div>"""

        items = []
        for i, m in enumerate(result.matches, 1):
            source_name = Path(m.source_chunk.source).name
            items.append(f"""
            <div class="match-item {m.level}">
                <div class="match-header">
                    <span class="badge {m.level}">{m.level.upper()}</span>
                    <span>相似度 {m.score:.4f}</span>
                    <span>·</span>
                    <span>来源: {esc(source_name)}</span>
                </div>
                <div class="pair">
                    <div class="side">
                        <div class="side-label">待查文档</div>
                        {esc(m.query_chunk.text)}
                    </div>
                    <div class="side">
                        <div class="side-label">知识库来源</div>
                        {esc(m.source_chunk.text)}
                    </div>
                </div>
            </div>""")

        return f"""
        <h2>🔍 相似片段（{len(result.matches)} 条）</h2>
        <div class="card">
            {"".join(items)}
        </div>"""

    # ----------------------------------------------------------------
    def _build_web_section(self, web_matches) -> str:
        if not web_matches:
            return ""

        items = []
        for i, m in enumerate(web_matches[:15], 1):
            items.append(f"""
            <div class="web-item">
                <div>
                    <span class="badge medium">score {m.score:.4f}</span>
                    <a href="{esc(m.url)}" target="_blank">{esc(m.title)}</a>
                </div>
                <div class="url">{esc(m.url)}</div>
                <div style="margin-top:6px;">{esc(m.body[:250])}</div>
            </div>""")

        return f"""
        <h2>🌐 网络原文定位（{len(web_matches)} 条）</h2>
        <div class="card">
            <p style="color:#7f8c8d;font-size:13px;margin-top:0;">
                提示: 网络搜索返回的是搜索引擎摘要，相似度天然偏低。
                以下结果用于人工复核参考，不构成抄袭判定。
            </p>
            {"".join(items)}
        </div>"""


# ==================================================================
# 测试
# ==================================================================
if __name__ == "__main__":
    from knowledge_base import KnowledgeBase
    from detector import PlagiarismDetector

    logger.info("加载知识库...")
    kb = KnowledgeBase(CFG)
    kb.load()

    logger.info("检测文档...")
    detector = PlagiarismDetector(CFG, kb)
    result = detector.detect("data/uploads/suspected.txt")

    if result:
        logger.info("生成报告...")
        gen = ReportGenerator(CFG)
        report_path = gen.generate(result)
        logger.info(f"报告: {report_path}")
        logger.info(f"用浏览器打开: file://{os.path.abspath(report_path)}")
