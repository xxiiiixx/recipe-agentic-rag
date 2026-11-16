# 30 天计划（2026-11-16 → 2026-12-15）

面向“Agent 偏算法”实习的作品集项目。目标岗位主投中型 AI 公司/垂直团队，一线大厂作冲刺；D14 简历冻结并开始投递，之后边投边补。

## 0. 已锁定的事实与口径

| 项 | 定稿 |
|---|---|
| 周期 | D1 = 2026-11-16（周一），D14 = 11-29 简历冻结，D15 = 11-30 开始投，D30 = 12-15 |
| 投入 | 25–30 h/周；60% 项目 / 25% 算法题与八股 / 15% 投递复盘 |
| 服务器 | Tesla T4 15GB，Driver 580.178.04，CUDA 13.0，仓库 `/workspace` |
| 环境 | conda `recipe-rag`，Python 3.12；torch 2.9.1 + cu126；sentence-transformers 6.3.0 |
| 模型 | 推理走魔搭 API（原生 function calling）；embedding 本地 `BAAI/bge-m3`（1024 维） |
| 主线 | 同一评测集上，先量化 RAG 底座变量，再量化 Agent 决策层相对静态 RAG 的增量与代价 |
| 统计 | 检索确定性单次 + 105 条 query 的 1000 次 bootstrap 95% CI；Agent 3 次中位数 |
| 执行 | 本地改代码 → 推分支 → 服务器 `git pull` → `scripts/run_matrix.py` → 结果推回 `results/` |

范围：P1 评测 + RAG 六组消融 + P2 工具层/状态机 + P3 规则版自适应 k + P5 基础 badcase。**P4 重排不做**；章节双通道召回为 stretch，延期第一个砍。

## 1. 环境初始化（D1）

```bash
cd /workspace
conda create -n recipe-rag python=3.12 -y
conda activate recipe-rag
pip install -U pip
pip install -r requirements.txt
pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu126
pip install "sentence-transformers==6.3.0"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
```

预期 `torch 2.9.1+cu126 True (7, 5)`。再跑一次真实 matmul，确认 T4 kernel 存在：

```bash
python -c "import torch; x=torch.randn(1024,1024,device='cuda'); print((x@x).sum().item())"
```

```bash
docker compose -f docker-compose.milvus.yml up -d
curl -f http://localhost:9091/healthz
find data/C8/cook -name '*.md' | wc -l     # 322
python scripts/selfcheck.py --root data/C8/cook --backend local --embedding hashing --dim 256
conda env export --from-history > environment.yml
pip freeze > requirements-lock.txt
```

模型选型在 D1 当天冻结：从魔搭控制台取 3 个支持 OpenAI 兼容 chat + function calling 的模型，用 `scripts/probe_models.py` 各跑 10 次 `search_recipes` 工具探针；合法调用 ≥9/10 且无持续 429 的候选里选最便宜的，备用模型写进 `.env` 注释。模型 ID 记入 `configs/server.yaml`，整月不换。

## 2. 逐日日历

### 第 1 周：地基 + 评测 + RAG 消融

| 日 | 日期 | 任务 | 产出 |
|---|---|---|---|
| D1 | 11-16 一 | conda 环境、torch/CUDA、Milvus、hashing 自检、模型选型 | 环境就绪；`environment.yml`、`requirements-lock.txt` |
| D2 | 11-17 二 | 修 P0：`stats` 崩溃、`--config` 位置、查询复用索引、local 索引隔离；补契约测试 | 可运行 `main`，tag `p0.1` |
| D3 | 11-18 三 | `${ENV}` 覆盖；`scripts/run_matrix.py` 骨架；3 组配置 smoke | runner 可跳跑、可续跑 |
| D4 | 11-19 四 | 评测集生成脚本 + LLM 造 120 条候选 | `eval/queries.candidates.jsonl` |
| D5 | 11-20 五 | 人工审核 105+15（第一批） | 审核一半；`eval/README.md` 标注规则 |
| D6 | 11-21 六 | 审完；服务器建 bge-m3 索引；跑 base + 分词 3 组 | 首批真实 dense 数字 |
| D7 | 11-22 日 | 跑完 RAG 12 个唯一配置；汇总 + 周复盘 | `results/rag/summary.md`；评测集 v1 冻结 |

### 第 2 周：Agent 层 + 首次冻结

| 日 | 日期 | 任务 | 产出 |
|---|---|---|---|
| D8 | 11-23 一 | 四个工具 + 结构化证据 + 单测 | `tools/` 可独立调用 |
| D9 | 11-24 二 | LangGraph 状态机骨架；魔搭 function calling 适配 | `agent/` 单轮跑通 |
| D10 | 11-25 三 | planner、证据化反思（≤2 轮）、规则版自适应 k | 三条决策规则可审计 |
| D11 | 11-26 四 | 10 条 query smoke，修工具参数/空结果/超轮次 | smoke 报告 |
| D12 | 11-27 五 | 全量 vs 静态两臂 ×105×3（后台，盯额度） | `results/agent/` 原始结果 |
| D13 | 11-28 六 | Agent 结果分析；badcase 分类 v1 | 增量/代价表 |
| D14 | 11-29 日 | 简历 v1 冻结；模拟面试 1；周复盘 | 简历 v1 + 10 道追问记录 |

### 第 3 周：投递 + 中间臂 + 深入

| 日 | 日期 | 任务 |
|---|---|---|
| D15 | 11-30 一 | 投 15 家（内推 2、Boss 8、官网/牛客 5）；Agent 中间两臂开跑 |
| D16 | 12-01 二 | 章节双通道 stretch；否则补 Agent 中间臂 |
| D17 | 12-02 三 | P5 trace 字段与汇总 |
| D18 | 12-03 四 | badcase 逐条归因 |
| D19 | 12-04 五 | 对照实验补跑；结果冻结 v1 |
| D20 | 12-05 六 | 报告提纲 + 图表初稿 |
| D21 | 12-06 日 | 模拟面试 2；周复盘；投递复盘 |

### 第 4 周：报告、二版简历、面试

| 日 | 日期 | 任务 |
|---|---|---|
| D22 | 12-07 一 | 技术报告上半（问题、设计、评测协议） |
| D23 | 12-08 二 | 技术报告下半（结果、消融、失败分析） |
| D24 | 12-09 三 | 简历 v2（补 Agent 真实数字）；投第二轮 |
| D25 | 12-10 四 | 博客 1：检索底座重构 |
| D26 | 12-11 五 | 博客 2：Agentic 检索决策与代价 |
| D27 | 12-12 六 | 最终补跑、仓库清理、tag `v1.0` |
| D28 | 12-13 日 | 模拟面试 3；被问倒清单归档 |
| D29 | 12-14 一 | 再投 15 家；完整彩排 |
| D30 | 12-15 二 | 交付检查 + 月度复盘 |

## 3. 实验矩阵

### RAG：6 组、12 个唯一配置

| 组 | 变量 | 配置 |
|---|---|---|
| A 分词 | `retrieval.tokenizer` | base / whitespace / jieba |
| B 臂 | `retrieval.arms` | base / dense / sparse |
| C 过滤 | `retrieval.filter_mode` | base(pre) / post |
| D 父块 | `retrieval.parent_strategy` | base(rrf_sum) / hit_count / max_score |
| E RRF | `retrieval.rrf_k` | base(60) / 10 / 30 / 100 |
| F 上下文 | `context.on_overflow` | base(truncate) / drop |

### Agent：4 臂

| 臂 | 组成 |
|---|---|
| G0 | 单次检索，k=5 |
| G1 | G0 + 证据化反思（≤2 轮） |
| G2 | G0 + 规则版自适应 k |
| G3 | G1 + G2 |

先跑 G0/G3，额度允许再补 G1/G2。

**指标**：Recall@5/10、MRR、nDCG@10、候选存活率、约束合规率、拒答准确率、Citation Precision、Faithfulness、Answer Relevancy、tokens/query、p50/p95、工具调用次数、`degraded` 比例。

## 4. 评测集规则

- 105 条正式 + 15 条 dev：单跳事实 35、约束组合 25、多食材 20、对比/多跳 15、该拒答 10。
- gold 标到菜名（`parent_id`），每条 1–5 个代表菜，不标 chunk；约束查询另记 `SearchFilter` 字段。
- `陈皮排骨汤` 两份按 `parent_id` 去重；同一 query 可多 gold，Recall 按集合算。
- LLM 批量造候选，100% 人工审（4–6 h）；不合格当场改或删。
- 该拒答题 gold 为空，记录 `expected_reason`；答了具体菜谱即判错。

## 5. 简历口径

### D14（只写真话）

- 重构 3.2k 行中文菜谱 RAG 检索底座：确定性块 ID、过滤下推、可切换分词/检索臂；在 105 条自建评测集上，稀疏臂 Hit@1 从 0% 提升到 83%，强约束查询合规率从 40% 提升到 100%，候选从 1701 降到 164。
- 搭建五类 120 条评测集与六组消融协议（bootstrap 95% CI），覆盖 Recall/MRR/nDCG、候选存活率、tokens 与 p50/p95。
- 实现工具层与 LangGraph 状态机（understand→plan→act→reflect→answer），支持 ≤2 轮证据化反思与规则版自适应 k；评估进行中。

### D24（补真实数字）

- 同一评测集上对比静态 RAG 与 Agentic 检索：约束类查询 Recall@5 +X pp，tokens +Y%，p95 +Z ms；识别 N 类 Agent 反而更差的查询并给出边界。
- 完成 12 配置 RAG 消融：量化分词、过滤下推、父块聚合、RRF k、上下文截断的独立贡献，结论附置信区间。

数字以服务器结果为准；D14 版本不写未跑出的数。

## 6. 风险与砍线

| 风险 | 触发信号 | 处理 |
|---|---|---|
| CUDA/torch 失败 | matmul 报 no kernel | 限时 2 h，退 CPU embedding；不影响其余消融 |
| Docker/Milvus 起不来 | healthz 不通 | 全程 `store.backend: local`，报告注明 |
| 魔搭额度/限流 | 10 条 smoke 429 | LLM 结果按 `(model,prompt,temp)` 缓存；重复降到 2 次；再砍 G1/G2 |
| 评测集审核超时 | D5 未审完 | 砍到 90+15，保持五类比例；不砍拒答题 |
| 进度延期 | D14 未完成 RAG 矩阵 | 砍双通道 → 砍 Agent 中间臂 → 砍博客 2 |
| 投递无反馈 | D21 仍无面试 | 重写简历 bullet，扩到 30 家/周 |

**绝不砍**：P1 评测集、RAG 主矩阵、D14 冻结、D15 起每天投递。

## 7. 交付物

技术报告（3000–5000 字）、博客两篇、D14/D24 两版简历、每周 15 分钟口头复盘、被问倒清单。
