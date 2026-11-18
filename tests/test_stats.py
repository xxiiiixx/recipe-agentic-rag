"""bootstrap 的契约：区间要么包住 0，要么给出显著的方向。

这组测试防的是最坏的一种错误：把噪声当成结论写进报告。
"""

from __future__ import annotations

import pytest

from rag_core.stats import format_ci, paired_bootstrap


def test_identical_series_has_zero_interval() -> None:
    a = [0.5, 0.8, 1.0, 0.2]
    diff, lo, hi = paired_bootstrap(a, a)
    assert diff == 0.0
    assert lo == 0.0 and hi == 0.0


def test_consistent_gain_excludes_zero() -> None:
    b = [0.0] * 50
    a = [1.0] * 50
    diff, lo, hi = paired_bootstrap(a, b)
    assert diff == pytest.approx(1.0)
    assert lo > 0


def test_noise_keeps_zero_inside_interval() -> None:
    a = [1.0, 0.0] * 25
    b = [0.0, 1.0] * 25
    diff, lo, hi = paired_bootstrap(a, b)
    assert diff == pytest.approx(0.0)
    assert lo <= 0 <= hi


def test_mismatched_length_raises() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])


def test_format_marks_non_significant() -> None:
    assert "(ns)" in format_ci(0.01, -0.05, 0.07)
    assert "(ns)" not in format_ci(0.05, 0.01, 0.09)
