# recipe-agentic-rag

把 [all-in-rag](https://github.com/datawhalechina/all-in-rag) 第 8 章（C8）的食谱 RAG 重写成**可消融的检索底座**，再在其上叠一层带检索规划与工具调用的 Agentic RAG。

底座沿用 C8 的父子块 + 混合检索；重写的是决定性 ID、可下推的过滤、可切换的检索臂，外加一套离线评估与 trace。

## 与原项目的区别

上游 C8 把流程串通了，但实现细节会让实验结论失真。这里只说三件最要紧的，**十条逐项对照见 [docs/01-vs-c8.md](docs/01-vs-c8.md)**：

- **所谓混合检索其实是纯向量检索**。上游 `BM25Retriever` 用 langchain 默认的 `text.split()` 分词，中文没空格，稀疏路实测命中率 0%。修成 `char_bigram` 后稀疏单臂 83% Hit@1，反超向量臂的 67%。
- **`chunk_id` 是 `uuid4()`，两次建库全变**，评测集只能退化到父文档粒度，父子块白做。本仓库用 `md5(parent_id ‖ 章节路径 ‖ 序号)`，索引可复现到块。
- **过滤发生在检索之后**（先取 `top_k*3` 再筛），条件越严召回越漏。改为过滤下推后，同一查询从「候选 1701 / 合规 40%」变成「候选 164 / 合规 100%」。

另外七处（融合身份用内容哈希会合并同名小节、分数写回共享 `metadata` 造成跨查询污染、按字符 `break` 整篇丢父文档、父块回填只数命中块个数、模板被 `rglob` 吃进索引、`FAISS` 进程内索引每次重建、LLM 与路径写死在代码里）均在 `01-vs-c8.md` 里给了位置、改法和依据。

## 架构

```mermaid
flowchart LR
    A[data/C8/cook<br/>322 篇菜谱] --> B[确定性切分]
    B --> C[Embedding] --> D[(Milvus<br/>向量 + INVERTED 索引)]
    B --> E[BM25<br/>可切换分词]
    F[SearchFilter] -->|下推| D
    F -->|候选集| E
    D --> G[RRF 融合]
    E --> G
    G --> H[父块回填] --> I[预算内截断] --> J[带引用生成]
```

`rag_core/` 不知道 Agent 存在；Agent 只能经工具 schema 调它；评估只读 trace。分层与接口承诺见 [docs/02-design.md](docs/02-design.md)。

## 快速开始

```bash
# 语料已随仓库提供：data/C8/cook 共 322 篇（来源、清洗与噪声见 data/README.md）
docker compose -f docker-compose.milvus.yml up -d       # milvus v2.5.14
curl -f http://localhost:9091/healthz

pip install -r requirements.txt
cp .env.example .env        # 填 LLM_API_KEY / EMBEDDING_API_KEY

python -m rag_core.cli build  --config configs/base.yaml
python -m rag_core.cli search "简单的素菜，10 分钟能做好" --max-difficulty 2
python -m rag_core.cli search "红烧肉要炖多久" --arms sparse   # 只跑稀疏臂
python -m rag_core.cli ask    "红烧肉要炖多久" --json

python scripts/selfcheck.py --config configs/base.yaml  # 五项自检 + 落盘报告
```

无 GPU：`embedding.provider` 换 `openai_compatible`；无 Docker：`store.backend` 换 `local`。
逐步验收与排错见 [docs/03-runbook.md](docs/03-runbook.md)。

### 自检输出（本地 hashing 向量，只验证机制）

| 检查 | 结果 |
|---|---|
| 索引确定性 | 两次切块 `chunk_id` / `parent_id` 逐条一致，1701 块无重复 |
| 语料洁净度 | 322 篇全为真菜谱，`template/示例菜` 已在取数时删除 |
| 分词（gold=菜名） | `char_bigram` 命中 92% Hit@1；`whitespace`（上游）0% |
| 过滤下推 | `pre` 合规率 100% / 平均候选 164；`post` 合规率 40% / 候选 1701 |

dense 臂数字要在服务器上换真 embedding 才有意义，届时替换上表。

## 目录

```
rag_core/       检索底座（14 模块）：切分 / 索引 / 双路召回 / RRF / 上下文 / 生成
data/C8/cook/   语料：322 篇菜谱（来源与清洗说明见 data/README.md）
scripts/        自检脚本：确定性 / 分词 / 检索臂 / 过滤下推 / 上下文
configs/        一份 YAML = 一次实验
docs/           01 与原项目的区别 · 02 设计（含 P1/P2 契约） · 03 跑通手册
```

`agent/`、`tools/`、`eval/` 的契约与评估协议已写在 [docs/02-design.md](docs/02-design.md) §5–§7，代码在 P1/P2 落地时新建。

## 消融矩阵

| 组 | 变量 | 臂 |
|---|---|---|
| A | 分词器 | `whitespace`（上游）/ `char_bigram` / `jieba` |
| B | 检索臂 | `dense` / `sparse` / 双路 |
| C | 过滤时机 | `pre`（下推）/ `post`（上游） |
| D | 父块排序 | `hit_count`（上游）/ `rrf_sum` / `max_score` |
| E | RRF 常数 | 10 / 30 / 60 / 100 |
| F | 上下文策略 | 整篇 break / 文档级截断 |

指标：Recall@k、MRR、nDCG@10、候选存活率、Faithfulness、Citation Precision，附 tokens/query 与 p50/p95 延迟。口径与报告格式见 [docs/02-design.md](docs/02-design.md) §7。

## 路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | 检索底座重构、确定性 ID、过滤下推、可切换臂、CLI | ✅ |
| P1 | 评测集（100 条）+ 指标脚本 + 六组消融报告 | ⬜ |
| P2 | LangGraph 状态机、检索工具化、反思与重试 | ⬜ |
| P3 | 自适应检索深度（按查询类型定 k） | ⬜ |
| P4 | 重排序 + 父子块粒度对比 | ⬜ |
| P5 | trace 与 badcase 归因 | ⬜ |

## 范围

不做多模态菜谱图、不做 GraphRAG（C9 已有）、不做 Web 产品——都会稀释"算法实验"这条主线。

语料与原始实现来自 [datawhalechina/all-in-rag](https://github.com/datawhalechina/all-in-rag)（MIT）。缺陷记录均为阅读源码所得。
