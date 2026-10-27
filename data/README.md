# 语料

322 篇中文菜谱（Markdown），随仓库提供，clone 后即可建库，不需要额外下载。

## 来源与许可

取自 [datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag) 的 `data/C8/cook`（MIT License），快照时间 2026-10-27，commit 见该仓库 `main`。菜谱正文本身来自社区贡献（上游 HowToCook 项目），引用与再分发请遵守上游说明。

## 相对上游做的三件事

1. **删掉 `dishes/template/`**：投稿模板「示例菜」，正文是"请在此处填写"式的填写说明，被 `rglob("*.md")` 当菜谱入库，宽泛查询会把它召回。
2. **只保留 `.md` 正文**：327 张配图（jpg/jpeg/png/webp，约 1 MB）未纳入仓库，检索链路只读 `.md`。
3. **删掉一个空目录** `drink/酸梅汤/imges`（上游拼写如此），git 不跟踪空目录。

## 结构

```
data/C8/cook/dishes/<category>/<菜名>.md     # 部分菜谱多一层同名目录
data/C8/cook/dishes/<category>/<菜名>/<菜名>.md
```

| 类目 | 篇数 | 类目 | 篇数 |
|---|---|---|---|
| meat_dish 荤菜 | 97 | soup 汤品 | 21 |
| vegetable_dish 素菜 | 54 | drink 饮品 | 21 |
| staple 主食 | 47 | dessert 甜品 | 17 |
| aquatic 水产 | 24 | semi-finished 半成品 | 10 |
| breakfast 早餐 | 22 | condiment 调料 | 9 |

## 已知噪声

- 168 篇正文里有 `![...](*.jpg)` 图片引用，图片已剔除，这些引用会原样进入切块文本。**是否剥离可以作为独立消融点**（它同时改变 dense 召回与送模型的 token 数，两个指标要分开看）。
- **同名重复**：`soup/陈皮排骨汤.md` 与 `soup/陈皮排骨汤/陈皮排骨汤.md` 是同一道菜的两份文件（322 篇 / 321 个菜名）。评测集若按菜名判金标，必须按 `parent_id` 去重或合并，否则一条 query 会同时命中两篇、Hit@1 直接失真。
- 难度标记不统一：大部分是 `预估烹饪难度：★★★★★`，少数缺失。缺失时 `parse_difficulty` 记 `(0, "未知")`，且不满足任何范围过滤——"未知"不该混进"简单"。
- 小部分菜谱只有 `# 菜名` 一行加零散段落，切块后不足 `min_chunk_chars` 的块会被丢弃。

## 更新语料

```bash
curl -L https://codeload.github.com/datawhalechina/all-in-rag/tar.gz/refs/heads/main -o /tmp/air.tgz
tar -xzf /tmp/air.tgz -C /tmp
rm -rf data/C8/cook && cp -r /tmp/all-in-rag-main/data/C8/cook data/C8/cook
rm -rf data/C8/cook/dishes/template
find data/C8/cook -type f ! -name '*.md' -delete
find data/C8/cook -name '*.md' | wc -l      # 322
```

语料变了，`selfcheck.py` 第一项的「语料洁净度」和块数会跟着变，实验报告要同步改。
