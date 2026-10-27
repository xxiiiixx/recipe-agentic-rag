"""Markdown 结构感知切分（自实现，替换 ``MarkdownHeaderTextSplitter``）。

为什么不用现成的切分器：
1. **ID 必须可复现**：上游在切分后给每块分配 ``uuid4()``，同一语料两次建库 ID 全变；
   自己切可以顺手生成 md5(父路径 + 章节路径 + 序号) 的稳定 ID。
2. **要能过滤贡献者注释**：Datawhale 菜谱里大量 ``<!-- ... -->`` 是写给投稿者的格式说明
   （"请在此处填写克数""图片放在同名目录"）。上游把它们原样切进块里、送进 embedding，
   等于给每个候选块注入同一段与菜品无关的噪声，还抬高了 BM25 的文档长度 |d|。
3. **章节类型要显式**：后面做"必需原料 vs 操作步骤"分通道召回时，
   需要知道这一块属于哪一级标题，而不是从字符串里再猜一次。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

_HEADER_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_FENCE_RE = re.compile(r"^\s*```")


@dataclass
class Section:
    """一个标题下的正文。``path`` 是从一级标题到当前标题的路径。"""

    path: List[str]
    body: str

    @property
    def path_str(self) -> str:
        return "/".join(p for p in self.path if p)

    @property
    def title(self) -> str:
        return self.path[-1] if self.path else ""


def strip_html_comments(text: str) -> str:
    return _HTML_COMMENT_RE.sub("", text)


def split_markdown(
    text: str,
    *,
    max_level: int = 3,
    drop_comments: bool = True,
    keep_header_in_body: bool = True,
) -> List[Section]:
    """按 ``#``/``##``/``###`` 切分，保留标题行在正文里。

    ``keep_header_in_body=True`` 与上游 ``strip_headers=False`` 的行为一致：
    块里带着"## 操作"，向量才分得清这是步骤还是原料清单。
    代码块（```）内部的行不会被当成标题。
    """
    if drop_comments:
        text = strip_html_comments(text)

    lines = text.splitlines()
    sections: List[Section] = []
    path: List[str] = []
    buffer: List[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        # 只有标题、没有正文的章节（如空小节）不产出块
        content_lines = [
            ln for ln in body.splitlines()
            if ln.strip() and not _HEADER_RE.match(ln.strip())
        ]
        if content_lines:
            sections.append(Section(path=list(path), body=body))
        buffer.clear()

    for line in lines:
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue
        match = None if in_fence else _HEADER_RE.match(line.strip())
        if match and len(match.group(1)) <= max_level:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            # 用层级重建路径：遇到更浅的标题就截断
            path = path[: level - 1]
            while len(path) < level - 1:
                path.append("")
            path.append(title)
            if keep_header_in_body:
                buffer.append(line)
            continue
        buffer.append(line)

    flush()
    return sections
