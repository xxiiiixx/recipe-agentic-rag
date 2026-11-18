"""统计工具：按 query 配对的 bootstrap 置信区间。

为什么配对：同一个 query 在不同臂上的难度差异远大于臂间的真实差异。
把 A、B 两臂的结果按 query 逐条相减再重采样，差值里已经消掉了"这条 query 本身难不难"，
比直接对两组均值做检验灵敏得多 —— 105 条 query 的量级上，这个差别足以决定
"看起来 +0.03" 到底是噪声还是信号。
"""

from __future__ import annotations

import random
import statistics
from typing import Sequence, Tuple


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """返回 (a-b 的均值差, CI 下界, CI 上界)。"""
    if len(a) != len(b):
        raise ValueError(f"配对样本长度不一致：{len(a)} vs {len(b)}")
    if not a:
        return 0.0, 0.0, 0.0
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    m = len(diffs)
    means = sorted(
        sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n)
    )
    lo = means[int(alpha / 2 * n)]
    hi = means[min(n - 1, int((1 - alpha / 2) * n))]
    return statistics.mean(diffs), lo, hi


def format_ci(diff: float, lo: float, hi: float) -> str:
    """CI 跨 0 时标 ns，避免把噪声读成结论。"""
    mark = "" if lo > 0 or hi < 0 else " (ns)"
    return f"{diff:+.3f} [{lo:+.3f}, {hi:+.3f}]{mark}"
