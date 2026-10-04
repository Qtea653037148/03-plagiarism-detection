"""
embedder.py
加载项目 2 训练的 BiEncoder，对文本编码。

为什么单独封装：
    - 项目 3 的 detector、knowledge_base 都要用
    - 编码接口要处理 batch、padding、GPU 传输等细节
    - 换模型时只改这一个文件
"""

import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from config import CFG
from utils import setup_logger

logger = setup_logger()


class Embedder:
    """
    句向量编码器。加载项目 2 微调的 BiEncoder。

    用法:
        embedder = Embedder(CFG)
        vecs = embedder.encode(["句子1", "句子2"])   # (2, D) numpy
    """

    def __init__(self, cfg=CFG, verbose=True):
        self.cfg = cfg
        self.device = cfg.device

        # ---------- 从项目 2 加载模型定义 ----------
        # 项目 2 的模型代码在另一个目录，这里临时加入 sys.path
        p2_dir = Path(cfg.biencoder_checkpoint).parent.parent
        if str(p2_dir) not in sys.path:
            sys.path.insert(0, str(p2_dir))

        from model import BiEncoderModel
        from transformers import AutoTokenizer

        # ---------- 加载 tokenizer ----------
        # 注意: 项目 3 的 Config 用的是 biencoder_model_name，
        # 项目 2 的 get_tokenizer(cfg) 期望 cfg.model_name，字段名不匹配。
        # 这里直接用 AutoTokenizer 加载，绕开跨项目的字段约定。
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.biencoder_model_name)

        # ---------- 加载模型 ----------
        checkpoint_path = cfg.biencoder_checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"找不到项目 2 的模型: {checkpoint_path}\n"
                f"请确认路径，或在 config.py 里改成正确的路径"
            )

        if verbose:
            logger.info(f"[Embedder] 加载 BiEncoder: {checkpoint_path}")

        model = BiEncoderModel(
            cfg.biencoder_model_name, pooling=cfg.biencoder_pooling
        ).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self.model = model

        if verbose:
            logger.info(
                f"[Embedder] 模型加载完成，"
                f"epoch={ckpt.get('epoch', '?')}, "
                f"val R@1={ckpt.get('recall@1', 0):.4f}"
            )

    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        把文本列表编码成 L2 归一化的向量矩阵。

        参数:
            texts:         文本列表
            batch_size:    批大小
            show_progress: 是否显示进度

        返回:
            (N, D) 的 float32 numpy 数组
        """
        if not texts:
            return np.zeros((0, self.model.hidden_size), dtype="float32")

        all_embs = []
        n = len(texts)
        iterator = range(0, n, batch_size)

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="编码")
            except ImportError:
                pass

        for i in iterator:
            batch = texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.cfg.max_seq_len,
                padding="max_length",
                return_tensors="pt",
            )
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            tti = enc.get("token_type_ids")
            if tti is not None:
                tti = tti.to(self.device)

            # 用 AMP 加速（推理时也有效）
            with torch.amp.autocast("cuda", enabled=self.cfg.device == "cuda"):
                emb = self.model(ids, mask, tti)

            all_embs.append(emb.float().cpu().numpy())

        result = np.concatenate(all_embs, axis=0).astype("float32")
        return result

    @torch.no_grad()
    def encode_one(self, text: str) -> np.ndarray:
        """编码单条文本，返回 (D,) 向量。"""
        return self.encode([text])[0]


# ==================================================================
# 测试
# ==================================================================
if __name__ == "__main__":
    logger.info("加载模型...")
    embedder = Embedder(CFG)

    texts = [
        "怎么办理信用卡",
        "如何申请信用卡",          # 与上句相似
        "今天天气怎么样",          # 不相似
        "今天的天气如何",          # 与上句相似
    ]
    vecs = embedder.encode(texts, show_progress=True)
    logger.info(f"编码完成: shape={vecs.shape}")

    # L2 范数应该全为 1
    norms = np.linalg.norm(vecs, axis=1)
    logger.info(f"L2 范数: {norms.round(4)}")

    # 相似度矩阵
    sim = vecs @ vecs.T
    logger.info("相似度矩阵:")
    for i in range(len(texts)):
        row = " ".join(f"{sim[i][j]:.3f}" for j in range(len(texts)))
        logger.info(f"  {texts[i][:15]:<20} {row}")

    logger.info("")
    logger.info("期望: 句0-句1 高相似，句2-句3 高相似，其余低")
