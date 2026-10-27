# 设计说明

"上游"指 `all-in-rag` 的 `code/C8/`。README 的缺陷表只列现象，这里记处理方式和理由。

## 1. 八处缺陷的处理

| 缺陷 | 处理 | 依据 |
|---|---|---|
| 模板入库 | 取数时删掉 `dishes/template`，代码不再背排除规则 | `template/示例菜` 含"请在此处填写"，会被宽泛查询召回；数据一次性清洗比在每个入口都排除更省事 |
| ID 不可复现 | `chunk_id = md5(parent_id ‖ section_path ‖ idx)` | 上游 uuid 让评测集只能退化到父文档粒度，父子块价值归零 |
| 难度只有字符串 | 同时存 `difficulty`(展示) 与 `difficulty_level`(过滤) | 未识别记 0 且不满足范围条件——"未知"不该混进"简单" |
| 过滤在检索之后 | `SearchFilter` 下推：Milvus `filter=expr` / 本地布尔掩码 | 上游取 `top_k*3` 后过滤，条件越严凑不满，且白花 3 倍算力 |
| 融合身份错误 | 按 `chunk_id` 融合 | 内容哈希会把不同菜品的同名小节（"洗净切块"）合并成一条 |
| 分数写回共享对象 | `ScoredChunk` 每次检索新建 | 上游 `metadata['rrf_score']` 让结果随查询顺序漂移 |
| 上下文整篇丢弃 | token 预算内**文档级截断**（边界回退） | 上游按字符 break，第 3 名超预算就整篇消失，抹平检索排序 |
| 父块回填 O(n²) | 字典聚合，保留 `rrf_score` | 上游对每个父 ID 全量扫一遍 `documents`，且丢弃融合分 |

**第八处（影响最大）**：`BM25Retriever.from_documents` 的分词是 langchain 默认的 `text.split()`。中文菜谱没有空格，整句被当成一个 token，稀疏路只在"查询与整段完全相等"时给分——实际退化成纯向量检索。`text.py` 的 `whitespace` 臂就是复现它，`char_bigram`（默认）与 `jieba` 是候选修法。

## 2. 为什么主后端是 Milvus

算法理由只有两条：

1. **过滤与向量检索在同一次调用完成**：先按 `expr` 求候选集、再在候选集上算距离，不存在"取 3 倍够不够"的赌注；
2. **标量字段建 INVERTED 索引**：`category_key` / `difficulty_level` / `parent_id` 各对应一类必用过滤。

工程上还有一条：`parent_id` 可用主键批查取代上游的全量扫描。

不换的理由也写清楚：本语料只有 ~10³ 块，**只做全量 Top-k 时 FAISS 更快**；Milvus 的收益是组合过滤下的召回与延迟稳定性。这个结论必须用 `filter_mode: pre/post` 的对照数说话，否则选型不严谨。

保留 numpy 后端（`LocalVectorStore`）是为了消融密度：起容器 + drop collection 会把单次迭代从 3s 拉到 30s，且评测要能在无 Docker 的机器上复现。两个后端走同一套接口和同一个 `SearchFilter`，差异只剩"谁执行过滤"。

## 3. 为什么 `rag_core` 不依赖 LangChain

不是反对它，而是实验变量要可控：`MarkdownHeaderTextSplitter` 给不出确定性 ID、`BM25Retriever` 把分词写死在默认函数里、组件升级会静默改变行为。自实现约 200 行，换来确定性 ID、可切换分词、候选子集检索、过滤模式开关。

LangChain/LangGraph 出现在 P2 的 `agent/` 层——那一层的价值是状态机与工具协议，不是文本切片。

## 4. 数据契约

`rag_core/schema.py` 是唯一契约，索引层、Milvus schema、评测集三处共用。

| 字段 | 类型 | 用途 |
|---|---|---|
| `chunk_id` | md5(32) | 主键、融合键、qrels 锚点 |
| `parent_id` | md5(32) | 父块回填、标量索引 |
| `dish_name` | str | 菜名精确匹配 |
| `category` / `category_key` | str | 中文展示 / ASCII 过滤（避开中文表达式转义） |
| `difficulty` / `difficulty_level` | str / int8 | 展示 / 范围过滤 |
| `section` / `section_path` | str | 章节类型，P3 分通道召回用 |
| `source_rel` | str | 回原文 |
| `text` | str | 向量化与 BM25 输入 |

## 5. 接口承诺（P2 之后不改）

```python
RecipeRAG.retrieve(query, flt=None, top_k=None, arms=None) -> RetrievalResult
RecipeRAG.ask(query, flt=None, top_k=None) -> (Answer, RetrievalResult)
RetrievalResult.to_trace() -> dict   # 落盘，供 eval/ 与 Agent trace 复用
```

Agent 层只能经这些入口触达检索，保证换编排框架后历史实验仍可比。

## 6. 已知未做

- `embed_query_prefix`（bge 查询指令）默认留空，未验证增益；
- RRF 常数 k 只做了配置化，未扫参（P1）；
- 父块回填固定取 `parent_top_n=2`，未按块覆盖度动态决定；
- embedding 模型横向对比与重排序一并在 P4 做。
