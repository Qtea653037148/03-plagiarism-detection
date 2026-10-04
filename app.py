"""
app.py
抄袭检测系统 - Streamlit Web UI

功能:
    - 侧边栏: 知识库管理（上传文档、查看统计、清空）
    - 主区域: 输入文本或上传文件 → 检测 → 可视化结果
    - 可选: 联网搜索原文定位

运行:
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
"""

import os
import time
import json
import tempfile
from pathlib import Path

import streamlit as st

from config import CFG
from knowledge_base import KnowledgeBase
from detector import PlagiarismDetector
from embedder import Embedder
from web_search import WebSearcher
from report import ReportGenerator, highlight_document


# ==================================================================
# 页面配置
# ==================================================================
st.set_page_config(
    page_title="抄袭检测系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.verdict-card {
    padding: 20px 28px;
    border-radius: 12px;
    font-size: 22px;
    font-weight: 700;
    margin: 16px 0;
    text-align: center;
}
.verdict-plagiarism {
    background: #ffebee; color: #c62828;
    border-left: 6px solid #c62828;
}
.verdict-clean {
    background: #e8f5e9; color: #2e7d32;
    border-left: 6px solid #2e7d32;
}
.verdict-warning {
    background: #fff3e0; color: #e65100;
    border-left: 6px solid #ef6c00;
}
.stat-box {
    background: #f8fafc;
    border-radius: 10px;
    padding: 16px 12px;
    text-align: center;
    border: 1px solid #e1e8ed;
}
.stat-value { font-size: 26px; font-weight: 700; color: #1e293b; }
.stat-label { font-size: 12px; color: #64748b; margin-top: 4px; }
.doc-view {
    background: #ffffff;
    border: 1px solid #e1e8ed;
    border-radius: 8px;
    padding: 20px;
    line-height: 2;
    font-size: 15px;
    white-space: pre-wrap;
    max-height: 500px;
    overflow-y: auto;
}
mark.hl-high {
    background: #ffcdd2;
    border-bottom: 2px solid #e74c3c;
    padding: 1px 3px;
}
mark.hl-medium {
    background: #ffe0b2;
    border-bottom: 2px solid #f39c12;
    padding: 1px 3px;
}
</style>
""", unsafe_allow_html=True)


# ==================================================================
# 资源加载
# ==================================================================
@st.cache_resource(show_spinner=False)
def get_embedder():
    return Embedder(CFG, verbose=False)


@st.cache_resource(show_spinner=False)
def get_web_searcher():
    return WebSearcher(embedder=get_embedder(), cfg=CFG)


def get_kb():
    if "kb" not in st.session_state:
        kb = KnowledgeBase(CFG, embedder=get_embedder())
        kb.load()
        st.session_state.kb = kb
    return st.session_state.kb


def get_detector():
    if "detector" not in st.session_state:
        st.session_state.detector = PlagiarismDetector(
            CFG, knowledge_base=get_kb()
        )
    return st.session_state.detector


# ==================================================================
# 侧边栏
# ==================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## 📚 知识库管理")

        kb = get_kb()
        stats = kb.stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("文档", stats["num_documents"])
        with col2:
            st.metric("文本块", stats["num_chunks"])

        st.caption(
            f"向量数: {stats['num_vectors']} · 维度: {stats['index_dim']}"
        )

        st.markdown("---")
        st.markdown("### ➕ 添加文档")
        st.caption("支持 PDF / DOCX / TXT，可多选")

        uploaded_files = st.file_uploader(
            "上传文件",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="kb_uploader",
        )

        if uploaded_files:
            if st.button("📥 导入知识库", type="primary", use_container_width=True):
                added = 0
                skipped = 0
                errors = []

                progress = st.progress(0, text="导入中...")
                for i, up_file in enumerate(uploaded_files):
                    progress.progress(
                        (i + 1) / len(uploaded_files),
                        text="处理 " + up_file.name,
                    )

                    try:
                        kb_dest = Path(CFG.kb_dir) / up_file.name
                        kb_dest.parent.mkdir(parents=True, exist_ok=True)
                        kb_dest.write_bytes(up_file.getbuffer())

                        ok = kb.add_document(str(kb_dest))
                        if ok:
                            added += 1
                        else:
                            skipped += 1
                    except Exception as e:
                        errors.append(up_file.name + ": " + str(e))

                progress.empty()

                if added > 0:
                    kb.save()

                if added:
                    st.success("✅ 成功导入 " + str(added) + " 个文档")
                if skipped:
                    st.info("⏭️ 跳过 " + str(skipped) + " 个（重复或解析失败）")
                for err in errors:
                    st.error(err)

                time.sleep(1)
                st.rerun()

        st.markdown("---")

        with st.expander("⚙️ 高级操作"):
            if st.button("🗑️ 清空知识库", use_container_width=True):
                kb.chunks = []
                kb.documents = []
                kb.index = None
                kb._seen_hashes = set()
                kb.save()

                kb_dir = Path(CFG.kb_dir)
                if kb_dir.exists():
                    for f in kb_dir.glob("*"):
                        if f.is_file():
                            f.unlink()

                st.session_state.pop("kb", None)
                st.session_state.pop("detector", None)
                st.success("知识库已清空")
                time.sleep(1)
                st.rerun()

            st.caption("知识库目录: " + CFG.kb_dir)
            st.caption("索引目录: " + CFG.index_dir)


# ==================================================================
# 主区域组件
# ==================================================================
def render_detection_form():
    st.markdown("### 📝 待查内容")

    input_mode = st.radio(
        "输入方式",
        ["✍️ 输入文本", "📎 上传文件"],
        horizontal=True,
        label_visibility="collapsed",
    )

    query_text = None
    query_name = "input_text"

    if input_mode == "✍️ 输入文本":
        default_text = (
            "自然语言处理是人工智能的重要分支。它研究如何让计算机理解和生成人类语言。\n"
            "北京是中国的首都，有着悠久的历史和丰富的文化资源。\n"
            "我今天下午去咖啡馆看了一本书，关于咖啡的历史。\n"
            "办理信用卡可以通过银行网点、官网或手机APP进行。"
        )
        query_text = st.text_area(
            "输入待检测文本",
            value=default_text,
            height=180,
            label_visibility="collapsed",
        )
    else:
        uploaded = st.file_uploader(
            "上传待查文档",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
            key="query_uploader",
        )
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as tmp:
                tmp.write(uploaded.getbuffer())
                query_text = tmp.name
            query_name = uploaded.name

    st.markdown("### ⚙️ 检测选项")

    col1, col2 = st.columns(2)
    with col1:
        enable_web = st.checkbox(
            "🌐 联网搜索原文来源",
            value=False,
            help="对高度相似的片段做网络搜索。会多花 5-15 秒。",
        )
    with col2:
        show_all = st.checkbox(
            "显示低度相似（score < 0.70）",
            value=False,
        )

    st.markdown("")
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        detect_btn = st.button(
            "🚀 开始检测",
            type="primary",
            use_container_width=True,
        )

    return query_text, query_name, enable_web, show_all, detect_btn


def render_verdict(result):
    if result.is_plagiarism():
        st.markdown(
            '<div class="verdict-card verdict-plagiarism">'
            '⚠️ 判定为抄袭 · 高度相似内容超过阈值</div>',
            unsafe_allow_html=True,
        )
    else:
        high = result.stats.get("high_count", 0)
        if high > 0:
            st.markdown(
                '<div class="verdict-card verdict-warning">'
                '🟡 疑似抄袭 · 存在高度相似片段，建议人工复核</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="verdict-card verdict-clean">'
                '✅ 未发现抄袭 · 未检测到高度相似内容</div>',
                unsafe_allow_html=True,
            )


def render_stats(result):
    s = result.stats
    cols = st.columns(6)
    metrics = [
        ("文档字数", "{:,}".format(s.get("total_chars", 0))),
        ("文本块数", s.get("total_chunks", 0)),
        ("高度相似", s.get("high_count", 0)),
        ("中度相似", s.get("medium_count", 0)),
        ("抄袭字符占比", "{:.1%}".format(s.get("char_ratio", 0))),
        ("检测耗时", "{:.2f}s".format(result.elapsed)),
    ]
    for col, item in zip(cols, metrics):
        with col:
            st.markdown(
                '<div class="stat-box">'
                '<div class="stat-value">' + str(item[1]) + '</div>'
                '<div class="stat-label">' + item[0] + '</div>'
                '</div>',
                unsafe_allow_html=True,
            )


def render_matches(result, show_all):
    matches = result.matches
    if not show_all:
        matches = [m for m in matches if m.level in ("high", "medium")]

    if not matches:
        st.info("没有需要展示的相似片段。")
        return

    st.markdown("### 🔍 相似片段（" + str(len(matches)) + " 条）")

    high = sum(1 for m in matches if m.level == "high")
    medium = sum(1 for m in matches if m.level == "medium")
    low = sum(1 for m in matches if m.level == "low")
    st.caption(
        "🔴 高度 " + str(high) + " 条 · "
        "🟠 中度 " + str(medium) + " 条 · "
        "⚪ 低度 " + str(low) + " 条"
    )

    emoji_map = {"high": "🔴", "medium": "🟠", "low": "⚪"}

    for i, m in enumerate(matches, 1):
        source_name = Path(m.source_chunk.source).name
        emoji = emoji_map.get(m.level, "⚪")

        st.markdown(
            "**" + str(i) + ". " + emoji + " " + m.level.upper() +
            "** · 相似度 **" + "{:.4f}".format(m.score) +
            "** · 来源: `" + source_name + "`"
        )

        col_left, col_right = st.columns(2)
        with col_left:
            st.caption("📝 待查文档")
            st.info(m.query_chunk.text)
        with col_right:
            st.caption("📄 知识库来源")
            st.success(m.source_chunk.text)

        st.markdown("---")


def render_source_distribution(result):
    dist = result.stats.get("source_distribution", {})
    if not dist:
        st.info("没有相似来源")
        return

    st.markdown("### 📚 相似来源分布")
    items = sorted(dist.items(), key=lambda x: -x[1]["count"])

    for name, info in items:
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            st.markdown("**📄 " + name + "**")
        with col2:
            st.markdown(
                "命中 **" + str(info["count"]) + "** 条 · **"
                + str(info["chars"]) + "** 字"
            )
        with col3:
            st.markdown(
                "最高相似度 **" + "{:.4f}".format(info["max_score"]) + "**"
            )


def render_highlighted_document(result):
    st.markdown("### 📄 原文高亮")
    highlighted = highlight_document(result.query_doc.text, result.matches)
    st.markdown(
        '<div class="doc-view">' + highlighted + '</div>',
        unsafe_allow_html=True,
    )
    st.caption("🔴 红色: 高度相似 · 🟠 橙色: 中度相似")


def render_web_results(web_matches):
    if not web_matches:
        st.info("未找到相关网络来源。")
        return

    st.markdown("### 🌐 网络原文定位（" + str(len(web_matches)) + " 条）")
    st.caption(
        "⚠️ 网络搜索返回的是搜索引擎摘要，相似度天然低于本地比对。"
        "以下结果仅供参考，不构成抄袭判定。"
    )

    filtered = [m for m in web_matches if m.score >= CFG.web_threshold_low]

    for i, m in enumerate(filtered[:10], 1):
        if m.score >= CFG.web_threshold_high:
            tag = "🟢 强相关"
        elif m.score >= CFG.web_threshold_medium:
            tag = "🟡 弱相关"
        else:
            tag = "⚪ 主题相关"

        st.markdown(
            "**" + str(i) + ". " + tag + "** · 相似度 "
            + "{:.4f}".format(m.score)
        )
        st.markdown("**" + m.title + "**")
        st.markdown("🔗 [" + m.url + "](" + m.url + ")")
        st.caption(m.body[:250])
        st.markdown("---")


def render_download_report(result, web_matches):
    st.markdown("---")
    st.markdown("### 💾 导出报告")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        try:
            gen = ReportGenerator(CFG)
            output_path = os.path.join(CFG.figures_dir, "report_latest.html")
            gen.generate(result, web_matches, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            st.download_button(
                "📥 下载 HTML 报告",
                data=html_content,
                file_name="plagiarism_report_" + str(int(time.time())) + ".html",
                mime="text/html",
                use_container_width=True,
            )
        except Exception as e:
            st.error("报告生成失败: " + str(e))

    with col2:
        json_data = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        st.download_button(
            "📥 下载 JSON 数据",
            data=json_data,
            file_name="plagiarism_result_" + str(int(time.time())) + ".json",
            mime="application/json",
            use_container_width=True,
        )


# ==================================================================
# 主流程
# ==================================================================
def main():
    render_sidebar()

    st.title("🔍 中文抄袭检测系统")
    st.markdown(
        "上传文档建立知识库，输入待查文本，检测**语义级抄袭**。"
        "支持本地知识库比对和联网原文定位。"
    )

    kb = get_kb()

    if kb.is_empty():
        st.warning(
            "⚠️ 知识库为空。请先在左侧上传文档建立知识库，再开始检测。"
        )

    st.markdown("---")

    query_text, query_name, enable_web, show_all, detect_btn = \
        render_detection_form()

    if detect_btn:
        if not query_text:
            st.error("请输入或上传待查内容")
            return

        if kb.is_empty():
            st.error("知识库为空，无法检测")
            return

        detector = get_detector()

        with st.spinner("🔍 检测中..."):
            if isinstance(query_text, str) and os.path.exists(query_text):
                result = detector.detect(query_text)
            else:
                result = detector.detect_text(
                    query_text, source_name=query_name
                )

        if result is None:
            st.error("检测失败，请检查输入")
            return

        st.markdown("---")

        render_verdict(result)
        render_stats(result)
        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 相似片段",
            "📚 来源分布",
            "📄 原文高亮",
            "🌐 网络搜索",
        ])

        with tab1:
            render_matches(result, show_all)

        with tab2:
            render_source_distribution(result)

        with tab3:
            render_highlighted_document(result)

        with tab4:
            if enable_web:
                with st.spinner("🌐 联网搜索中（约 10-20 秒）..."):
                    searcher = get_web_searcher()
                    web_matches = searcher.search_detection_result(
                        result, max_sentences=3
                    )
                render_web_results(web_matches)
            else:
                st.info(
                    "联网搜索未启用。勾选「🌐 联网搜索原文来源」后重新检测。"
                )
                web_matches = []

        render_download_report(result, web_matches)

    else:
        with st.expander("📖 使用说明", expanded=True):
            st.markdown("""
### 快速上手

**第一步：建立知识库**

左侧上传文档（PDF / DOCX / TXT），系统会自动：
1. 解析为纯文本
2. 按句切分成小块
3. 用 BiEncoder 编码成向量
4. 建立 FAISS 索引

**第二步：检测**

- 输入文本或上传文件
- 点「🚀 开始检测」

**第三步：查看结果**

- **判定卡片**：红色 = 抄袭，黄色 = 疑似，绿色 = 干净
- **相似片段**：并排显示待查和来源
- **原文高亮**：原文中被判定抄袭的句子标红
- **网络搜索**（可选）

### 检测原理

待查文档 → 分句 → BiEncoder 编码
→ FAISS 从知识库检索 Top-K
→ 按相似度分级
→ 文档级聚合判定

text

### 阈值说明

- 高度相似：score ≥ 0.85
- 中度相似：score ≥ 0.70
- 低度相似：score ≥ 0.55
- 文档级判定：高度相似块占比 ≥ 20%
            """)


if __name__ == "__main__":
    main()
