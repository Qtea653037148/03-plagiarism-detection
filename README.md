# 中文同语义抄袭检测系统

基于 BiEncoder 语义向量的抄袭检测系统。支持本地知识库比对、联网原文定位、可视化报告和 Web UI。

## 项目背景

传统查重基于字符串匹配，无法识别 **同义改写、语序调整、翻译回译** 等抄袭手段。本项目用语义相似度替代字面匹配，能检测“意思一样但写法不同”的抄袭。

**核心能力**:

- 本地知识库比对: 上传文档建立知识库，检测新文档与知识库的语义相似片段
- 联网原文定位: 对高度相似片段联网搜索，定位可能的原始出处
- 分级判定: 按相似度分 high / medium / low 三级，文档级聚合后给出抄袭结论
- 可视化报告: HTML 报告 + 原文高亮 + 来源分布
- Web UI: Streamlit 交互界面，支持上传、检测、导出

## 系统架构

```
待查文档 (PDF/DOCX/TXT)
    │
    ▼
① 文档解析 → 纯文本
② 分句 → 文本块
③ BiEncoder 编码 → 向量
    │
    ├─────────────┬─────────────────────┐
    ▼             ▼                     ▼
④ 本地检索    ⑤ 网络检索            未来扩展
(FAISS)       (必应中国)             引用数据库
    │             │
    └──────┬──────┘
           ▼
    ⑥ 分级 + 去重
           ▼
    ⑦ 文档级聚合判定
           ▼
    ⑧ 可视化报告 + Web UI
```

## 技术栈

| 模块 | 技术 |
|------|------|
| 句向量模型 | hfl/rbt3 + 对比学习微调 (来自项目 2) |
| 向量检索 | FAISS IndexFlatIP |
| 文档解析 | pdfplumber / pypdf / python-docx |
| 分块 | 正则分句 + 三种策略 (sentence/paragraph/sliding) |
| 网络搜索 | cn.bing.com + BeautifulSoup |
| Web UI | Streamlit |
| 报告 | HTML + 内联 CSS |

## 快速开始

### 环境

- python >= 3.10
- torch >= 2.5
- transformers == 4.44.2
- sentence-transformers >= 2.7
- faiss-cpu >= 1.7
- streamlit >= 1.30
- pypdf / pdfplumber / python-docx
- jieba / loguru

### 安装

```bash
pip install -r requirements.txt
```

### 前置条件

本项目的句向量模型来自 **项目 2**，需要先完成项目 2 的训练:

```
../02-semantic-search/checkpoints/biencoder_best.pt
```

如果路径不同，修改 `config.py` 里的 `biencoder_checkpoint`。

### 建立知识库

方式 1 - 命令行:

```bash
# 把文档放到 data/knowledge_base/
python knowledge_base.py
```

方式 2 - Web UI（推荐）:

启动 Web 界面后，在左侧上传文档即可。

### 检测

命令行:

```bash
python detector.py
```

Web UI（推荐）:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Windows 浏览器访问 `http://localhost:8501`。

### 生成报告

```bash
python report.py
# 生成 figures/report.html
```

## Web UI 使用

### 左侧: 知识库管理

- 查看当前知识库统计（文档数 / 块数 / 向量数）
- 上传文档 (PDF / DOCX / TXT，可多选)
- 导入后自动解析、分块、编码、建索引
- 一键清空重建

### 主区域: 检测

1. 输入文本 或 上传文件
2. 可选勾选“联网搜索原文来源”
3. 点“开始检测”

### 结果展示

- 判定卡片: 红色(抄袭) / 黄色(疑似) / 绿色(干净)
- 统计面板: 字数 / 块数 / 高/中相似数 / 抄袭占比 / 耗时
- 四个 Tab:
  - 相似片段: 并排展示待查与来源
  - 来源分布: 哪个文档被抄了多少
  - 原文高亮: 抄袭片段在原文中标红
  - 网络搜索: 联网定位的原文链接

### 导出

- HTML 报告: 带样式，可分享打印
- JSON 数据: 结构化结果

## 关键参数

```python
# 相似度阈值
threshold_high = 0.85      # 高度相似 (严重抄袭)
threshold_medium = 0.70    # 中度相似 (可疑)
threshold_low = 0.55       # 低度相似 (参考)

# 文档级判定
doc_plagiarism_ratio = 0.20   # 高度相似块占比 > 20% 判为抄袭

# 网络搜索阈值 (更低，因为返回的是摘要)
web_threshold_high = 0.60
web_threshold_medium = 0.45
web_threshold_low = 0.30

# 分块策略
chunk_strategy = "sentence"   # sentence / paragraph / sliding
min_chunk_len = 10            # 过滤太短的块

# 检索
recall_top_k = 20             # 召回数量
```

## 检测原理

### 为什么用语义相似度而不是字符串匹配

```
原文: "北京是中国的首都，有着悠久的历史"
抄袭: "北京作为中国的首都，历史十分悠久"

字符串匹配: 相似度 < 0.3   (无法识别)
语义匹配:   相似度 > 0.85  (成功识别)
```

BiEncoder 把句子编码成 768 维向量，语义相近的句子在向量空间中距离近。

### 分级判定逻辑

```
每个待查句 → 与知识库所有句比对 → 得到相似度分数

score >= 0.85  → high     (几乎确定抄袭)
score >= 0.70  → medium   (可能改写或引用)
score >= 0.55  → low      (只是主题相关)
score < 0.55   → 忽略

文档级判定:
    high_count / total_chunks >= 20%  →  判定为抄袭
```

### 网络搜索的定位

搜索引擎返回的是 **摘要**（100-200 字），不是完整网页原文。因此:

- 相似度天然低于本地比对
- 定位是“给人工复核方向”，不构成自动判定
- 想更准需要抓取网页正文（本项目未做，见改进方向）

## 项目结构

```
03-plagiarism-detection/
├── config.py              # 全局配置
├── utils.py               # 日志/文本清洗/哈希
├── parser.py              # 文档解析 (PDF/DOCX/TXT)
├── chunker.py             # 分句/分块
├── embedder.py            # 加载项目2的BiEncoder
├── knowledge_base.py      # 知识库管理 + FAISS
├── detector.py            # 抄袭检测核心
├── web_search.py          # 网络原文定位
├── report.py              # HTML报告生成
├── app.py                 # Streamlit Web UI
├── font_utils.py
├── requirements.txt
├── data/
│   ├── knowledge_base/    # 知识库文档
│   ├── uploads/           # 待查文档
│   └── index/             # FAISS索引
├── figures/               # 报告输出
└── README.md
```

## 实测效果

测试场景: 知识库有 3 个文档（NLP 介绍 / 信用卡 / 北京），待查文档 8 句话，其中 6 句从知识库照抄。

结果:

| 指标 | 值 |
|------|-----|
| 检测耗时 | 1.1s |
| 高度相似 | 6 条 |
| 中度相似 | 0 条 |
| 低度相似 | 2 条 |
| 抄袭字符占比 | 72.6% |
| 判定 | 抄袭 |

完全照抄的片段相似度 = 1.0000；跨主题的原创句子不会被误检。

## 局限性

### 当前局限

1. 网络搜索准确度有限: 只比对搜索引擎摘要，不是网页正文
2. 长文本处理: 单块超过 max_seq_len=128 会被截断
3. PDF 解析: 双栏排版、扫描件效果差
4. 改写识别: 深度改写（段落重组）可能漏检
5. 大规模知识库: IndexFlatIP 是暴力搜索，10 万+ 文档需要换 IVF / HNSW

### 改进方向

- [ ] 网络搜索抓取网页正文，做精确比对
- [ ] 换用更大的句向量模型 (BGE-large-zh)
- [ ] CrossEncoder 精排模块 (召回 Top-20 → 精排 Top-5)
- [ ] 段落级对齐 (把相似的句子聚合为段落，识别整段抄袭)
- [ ] 引用识别 (排除带引号的合理引用)
- [ ] 支持更多的文档格式 (Markdown / HTML / EPUB)
- [ ] 部署为 FastAPI 服务
- [ ] 增量索引 (增量添加文档不需要重建整个索引)

## 与前两个项目的关系

本项目是“中文 NLP 全链路”系列的第 3 个项目:

| 项目 | 核心 | 技术 |
|------|------|------|
| 项目 1: Transformer 情感分类 | 从零实现 Encoder | MHA / 位置编码 / 训练循环 |
| 项目 2: 语义搜索 | 句向量训练 | 对比学习 / BiEncoder / FAISS |
| 项目 3: 抄袭检测（本项目） | 落地应用 | 文档解析 / 召回+判定 / Web UI |

项目 3 直接复用项目 2 训练的 BiEncoder 模型。

## 参考

- [Sentence-BERT](https://arxiv.org/abs/1908.10084)
- [FAISS](https://github.com/facebookresearch/faiss)
- [LCQMC 数据集](http://icrc.hitsz.edu.cn/Article/show/171.html)
- [Streamlit](https://streamlit.io/)

## License

MIT License
