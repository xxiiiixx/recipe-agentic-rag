# recipe-agentic-rag

把 [all-in-rag](https://github.com/datawhalechina/all-in-rag) 第 8 章的食谱 RAG 重写成**可消融的检索底座**，再在其上叠加带检索规划与工具调用的 Agentic RAG。

## 核心结论

| 结论 | 数字 |
|---|---|
| 上游“混合检索”实际是纯向量 | 修好分词后，稀疏臂 Hit@1 0% → 83%；`char_bigram` 92% |
| 过滤下推 | 候选 1701 → 164，约束查询合规率 40% → 100% |
| 确定性索引 | 322 篇 → 1701 块，两次建库 `chunk_id` 完全一致 |
| 查询解析 + 难度重排 | compare .938 → 1.000、single_hop .943 → 1.000；constraint R@5 .563 → .633 |
| Agent 编排 | 多路检索只剩 multi_hop 收益（R@10 .071 → .214）；反思环无收益，如实记录 |

前三行来自 `scripts/selfcheck.py`（hashing 向量 + local 后端，只验证机制）；评测集与 Agent 的数字来自 `scripts/eval_retrieval.py` 与 `scripts/eval_agent.py`（105 test，bge-m3 + Milvus Lite）。

## 架构

```text
Markdown → 按章节切块 → Milvus 向量 + BM25 → RRF 融合 → 父块回填 → 预算内截断 → 带引用生成
                                                        ↑
Agent: understand → plan → act → reflect → answer（工具层只经 schema 调用检索）
```

`rag_core/` 不知道 Agent 存在；Agent 只能经工具 schema 调用它；评估只读 trace。

## 快速开始

```bash
docker compose -f docker-compose.milvus.yml up -d
curl -f http://localhost:9091/healthz
pip install -r requirements.txt
pip install -r requirements-embedding.txt
cp .env.example .env          # 填 LLM_API_KEY（ask 需要）
python -m rag_core.cli --config configs/server.yaml build
python -m rag_core.cli --config configs/server.yaml search "红烧肉要炖多久"
python scripts/selfcheck.py --root data/C8/cook --backend local --embedding hashing --dim 256
```

无 Docker 用 `store.backend: local`；无 GPU 用云端 embedding。CUDA 机器按 [docs/runbook.md](docs/runbook.md) 的 cu126 命令装 torch。

## 目录

```text
rag_core/    检索底座：切分 / 索引 / 双路召回 / RRF / 上下文 / 生成
rag_core/agent/  P2：工具层 / 查询理解 / 状态机
data/C8/     322 篇中文菜谱
eval/        105 test + 15 dev 评测集与标注规则
configs/     一份 YAML = 一次实验；server.yaml 继承 base.yaml
scripts/     selfcheck / 检索与生成评测 / Agent 对照评测
tests/       契约测试
docs/        与上游差异 / 设计 / 跑通手册 / Agent 设计
```

## 文档

- [设计与评测协议](docs/design.md)
- [Agent 层设计](docs/agent.md)
- [跑通与排错](docs/runbook.md)

## 路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | 检索底座、确定性 ID、过滤下推、可切换臂 | ✅ |
| P1 | 105 条评测集 + 六组消融 | ✅ |
| P2 | 工具层 + 状态机 + 反思 | ✅ |
| P3 | 规则版自适应检索深度 | 计划 |
| P5 | trace 与 badcase 归因 | 计划 |

P4 重排不做：组件成熟、对 T4 收益低。
