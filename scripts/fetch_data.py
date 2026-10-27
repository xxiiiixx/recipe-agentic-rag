"""拉取 C8 语料（323 篇菜谱 Markdown），只依赖标准库。

仓库不内联菜谱数据，第一次跑之前执行：

    python scripts/fetch_data.py                # -> data/C8/cook/dishes/<category>/...
    python scripts/fetch_data.py --dest data/C8/cook --ref main

数据来自 datawhalechina/all-in-rag 的 ``data/C8/cook``，流式解包，不落中间 tar。
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

URL = "https://codeload.github.com/datawhalechina/all-in-rag/tar.gz/refs/heads/{ref}"
PREFIX = "data/C8/cook/"


def fetch(dest: Path, ref: str) -> Counter:
    dest.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    with urllib.request.urlopen(URL.format(ref=ref), timeout=120) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                # 成员名形如 all-in-rag-main/data/C8/cook/dishes/aquatic/....md
                idx = member.name.find(PREFIX)
                if idx < 0 or not member.name.endswith(".md"):
                    continue
                rel = member.name[idx + len(PREFIX):]
                if not rel or not rel.endswith(".md"):
                    continue
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                target.write_bytes(handle.read())
                parts = Path(rel).parts
                if parts[0] == "dishes" and len(parts) > 1:
                    stats[parts[1]] += 1
                else:
                    stats[parts[0] if len(parts) > 1 else "."] += 1
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 C8 菜谱语料")
    parser.add_argument("--dest", default="data/C8/cook")
    parser.add_argument("--ref", default="main")
    args = parser.parse_args()

    dest = Path(args.dest)
    stats = fetch(dest, args.ref)
    total = sum(stats.values())
    if not total:
        print("没有取到任何 Markdown，检查网络或上游路径是否变化", file=sys.stderr)
        return 1

    print(f"已下载 {total} 篇菜谱 -> {dest}")
    for name, count in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<16} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
