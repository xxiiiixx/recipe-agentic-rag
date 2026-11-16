# 设计

同一份语料、沿用父子块与混合检索，但把检索底座重写成可消融、可复现的实现；Agent 层只经工具 schema 触达检索。上游指 [all-in-rag](https://github.com/datawhalechina/all-in-rag) 的 `code/C8/`。

## 1. 与上游的关键差异

| 上游 | 本仓库 | 为什么 |
|---|---|---|
| BM25 用 `text.split()` | `char_bigram` / `jieba` / `whitespace` 三臂 | 中文无空格，上游稀疏路实测 0% |
| `chunk_id = uuid4()` | `md5(parent_id, section_path, index)` | 两次建库 ID 不变，qrels 可锚定 |
| 先取 `top_k*3` 再过滤 | `SearchFilter` 下推 | 条件越严召回越差，且白算 3 倍 |
| 父块按命中块数排序 | `rrf_sum` / `hit_count` / `max_score` | 个数排序丢弃融合分 |
| 超预算 `break` 整篇 | token 预算内文档级截断 | 保排序；`drop` 可复现上游 |
| FAISS 进程内索引 | Milvus + local numpy | 过滤与检索同次调用；无 Docker 可复现 |

另外四处已在实现中移除：内容哈希做融合身份会合并同名小节、分数写回共享 `metadata` 造成跨查询污染、`rglob` 把投稿模板吃进索引、LLM 与数据路径写死。细节见 git 历史与 `tests/`。

## 2. 模块

| 模块 | 职责 |
|---|---|
| `schema.py` | 数据契约：`Recipe` / `Chunk` / `SearchFilter` / `ScoredChunk` / `RetrievalResult` |
| `text.py` | 分词三臂、token 估算、预算内截断 |
| `bm25.py` | 倒排索引 + BM25，支持候选子集内检索 |
| `markdown_splitter.py` | 按 `#/##/###` 切分，忽略代码围栏，剥离 HTML 注释 |
| `data_preparation.py` | 语料加载、确定性 ID、父子块、父块排序 |
| `embeddings.py` | `sentence_transformers` / `openai_compatible` / `hashing` |
| `vector_store.py` | Milvus 与 local 两套后端，共用同一 `SearchFilter` |
| `retrieval.py` | 双路召回、RRF 融合、父块回填、trace |
| `context.py` | token 预算内组装，截断或丢弃 |
| `generation.py` | 提示词、引用标注、`citation_precision` |
| `config.py` | YAML → dataclass，支持 `extends` 与 `${ENV}` |
| `pipeline.py` / `cli.py` | 门面与命令行 |

## 3. 数据契约

- `chunk_id` / `parent_id`：md5，稳定主键，融合与 qrels 的锚点。
- `dish_name` / `category` / `category_key`：中文展示与 ASCII 过滤分离。
- `difficulty` / `difficulty_level`：展示与范围过滤分离，`0` 表示未知，不参与范围过滤。
- `section` / `section_path`：章节类型与完整路径，P3 分通道召回用。
- `text`：向量与 BM25 的共同输入。

## 4. 接口承诺

```python
RecipeRAG.retrieve(query, flt=None, top_k=None, arms=None) -> RetrievalResult
RecipeRAG.ask(query, flt=None, top_k=None) -> (Answer, RetrievalResult)
RetrievalResult.to_trace() -> dict
```

`rag_core/` 不知道 Agent 存在；Agent 只能经这些入口触达检索；评估只读 trace。

## 5. P2 工具与状态机（契约，代码待落地）

| 工具 | 参数 | 用途 |
|---|---|---|
| `search_recipes` | `query`, `top_k`, `filters` | 主检索（双路 + RRF） |
| `filter_by_constraints` | `filters` | 只筛条件，返回候选规模 |
| `get_recipe_detail` | `dish_name`, `section` | 取整篇或指定章节 |
| `compare_recipes` | `dishes[]` | 多菜对比 |

状态机：`understand → plan → act → reflect → answer`。

- 反思只在空结果、最高融合分低于阈值、命中违反约束时触发，最多 2 轮；超限直接出答案并标 `degraded=True`。
- 工具返回结构化证据（`score`、来源臂、候选数），空结果不是异常。
- 同 `query + config` 两次运行，工具调用序列一致（温度 0.1，允许文本差异）。

## 6. 评估协议（P1）

105 条正式 + 15 条 dev：单跳 35、约束 25、多食材 20、对比/多跳 15、该拒答 10。gold 标到菜名（`parent_id`），不标 chunk；`陈皮排骨汤` 两份按 `parent_id` 去重。

检索指标为确定性单次 + 1000 次 query bootstrap 95% CI；Agent 指标为 3 次中位数。指标：Recall@5/10、MRR、nDCG@10、候选存活率、约束合规率、拒答准确率、Citation Precision、Faithfulness、Answer Relevancy、tokens/query、p50/p95。

每次只改一个变量；一个配置 = 一个 YAML；结果与配置同目录归档。

## 7. 语料与配置

语料来源、清洗规则与已知噪声见 [data/README.md](../data/README.md)；配置的继承与 `.env` 分工见 [configs/README.md](../configs/README.md)。
