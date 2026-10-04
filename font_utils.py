"""
font_utils.py
统一配置 matplotlib 中文字体。所有画图脚本在 import matplotlib 后调用 setup_fonts()。
找不到中文字体时自动回退到英文标签，不会报错。
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams


# 常见中文字体候选，按优先级排列
CJK_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
    "Source Han Sans CN", "Source Han Sans SC",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
    "SimHei", "Microsoft YaHei", "PingFang SC", "Heiti SC", "STHeiti",
]


def setup_fonts(verbose: bool = True) -> str | None:
    """
    扫描系统已装字体，选第一个匹配的中文字体。
    返回选中字体名，找不到返回 None。
    """
    available = {f.name for f in fm.fontManager.ttflist}

    for name in CJK_CANDIDATES:
        if name in available:
            rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            rcParams["axes.unicode_minus"] = False  # 负号正常显示
            if verbose:
                print(f"[Font] 使用中文字体: {name}")
            return name

    # 没找到，回退英文标签，不报错
    rcParams["axes.unicode_minus"] = False
    if verbose:
        print("[Font] 未找到中文字体，图中文字将用英文替代")
        print("[Font] 建议: sudo apt install fonts-noto-cjk")
    return None
