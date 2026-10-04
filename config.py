"""
config.py
项目 3 全局配置。
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ==================== 路径 ====================
    project_root: str = "."
    kb_dir: str = "data/knowledge_base"       # 知识库原始文档
    upload_dir: str = "data/uploads"          # 用户上传
    index_dir: str = "data/index"             # FAISS 索引
    figures_dir: str = "figures"              # 报告图片

    # ==================== 模型 ====================
    # 复用项目 2 微调的 BiEncoder，但也支持直接加载项目 2 的 checkpoint
    biencoder_checkpoint: str = "../02-semantic-search/checkpoints/biencoder_best.pt"
    biencoder_model_name: str = "hfl/rbt3"
    biencoder_pooling: str = "mean"

    # CrossEncoder 用于精排（可选，稍后加）
    crossencoder_checkpoint: str = "../02-semantic-search/checkpoints/crossencoder_best.pt"
    crossencoder_model_name: str = "hfl/rbt3"

    max_seq_len: int = 128           # 长句场景，比项目 2 的 48 大

    # ==================== 分块 ====================
    chunk_strategy: str = "sentence"    # sentence / paragraph / sliding
    sliding_window: int = 3             # 滑窗大小（句子数）
    sliding_stride: int = 1             # 滑窗步长
    min_chunk_len: int = 10             # 过滤太短的块（无意义）

    # ==================== 检索 ====================
    recall_top_k: int = 20              # BiEncoder 召回数量
    rerank_top_k: int = 5               # 精排后返回数量

    # ==================== 抄袭判定 ====================
    # 相似度阈值（分级）
    threshold_high: float = 0.85        # 高度相似（严重抄袭）
    threshold_medium: float = 0.70      # 中度相似（可疑）
    threshold_low: float = 0.55         # 低度相似（参考）

    # 文档级阈值
    doc_plagiarism_ratio: float = 0.20  # 高相似内容占比 > 20% 判定为抄袭

    # ==================== 网络检索 ====================
    enable_web_search: bool = True
    web_search_provider: str = "duckduckgo"    # duckduckgo / bing / serpapi
    web_search_max_results: int = 10           # 每次查询返回结果数
    web_search_timeout: int = 10               # 秒
    web_search_top_k_sentences: int = 3        # 只对 top-K 相似句做网络检索

    # 网络搜索相似度阈值（比本地低，因为搜索引擎返回的是摘要）
    web_threshold_high: float = 0.60           # 强相关
    web_threshold_medium: float = 0.45         # 弱相关
    web_threshold_low: float = 0.30            # 只是主题相关

    # 网络搜索相似度阈值（比本地低，因为搜索引擎返回的是摘要）
    web_threshold_high: float = 0.60           # 强相关
    web_threshold_medium: float = 0.45         # 弱相关
    web_threshold_low: float = 0.30            # 只是主题相关

    # ==================== 设备 ====================
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self):
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print("[Config] CUDA 不可用，切换到 CPU")


CFG = Config()


if __name__ == "__main__":
    import torch
    print(CFG)
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
