# 服务器跑通手册（P0 验证）

从零到"检索能跑、指标能看"的一条路径。每一步都有可验证的输出，任何一步失败都停在那一步，不要往下滚。

## 0. 前置

| 项 | 要求 | 检查命令 |
| --- | --- | --- |
| Python | 3.10+ | `python -V` |
| Docker | 仅 Milvus 后端需要 | `docker info` |
| 磁盘 | 语料 20MB，模型 100–500MB | `df -h .` |
| 网络 | 拉 codeload / PyPI / 模型权重 | `curl -I https://codeload.github.com` |

## 1. 装依赖

```bash
git clone https://github.com/xxiiiixx/recipe-agentic-rag.git && cd recipe-agentic-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 不含 torch。用本地 BGE 再装：

```bash
pip install -r requirements-embedding.txt
```

## 2. 起 Milvus

```bash
docker compose -f docker-compose.milvus.yml up -d
curl -f http://localhost:9091/healthz   # 等到返回 OK
docker compose -f docker-compose.milvus.yml ps
```

三个容器（etcd / minio / milvus-standalone）都在 `running` 才算好。端口冲突就改 compose 里的映射，同时改配置 `store.uri`。

## 3. 取语料

```bash
curl -L https://codeload.github.com/datawhalechina/all-in-rag/tar.gz/refs/heads/main -o /tmp/air.tgz
tar -xzf /tmp/air.tgz -C /tmp && mkdir -p data/C8
cp -r /tmp/all-in-rag-main/data/C8/cook data/C8/
rm -rf data/C8/cook/dishes/template          # 投稿模板，不是菜谱
find data/C8/cook -name '*.md' | wc -l       # 322
```

语料不入库（`.gitignore` 里有 `data/`），这四行是唯一的数据准备步骤。**清洗只做一次**：`template/示例菜` 删掉之后，代码里不再需要任何排除规则；重新取数时记得再删一次。

## 4. 选 embedding

三选一，决定 `configs/` 里用哪份配置。

**A. 本地 BGE（默认，`configs/base.yaml`）**：`BAAI/bge-small-zh-v1.5`，512 维，首次运行会下载权重。

**B. 云端 embedding（无 GPU 时更稳）**：

```bash
cat > configs/local-api.yaml <<'YAML'
name: api
workspace: artifacts
dataset: {root: data/C8/cook}
embedding:
  provider: openai_compatible
  model: BAAI/bge-m3
  dim: 1024
  base_url: https://api.siliconflow.cn/v1
  api_key: ${EMBEDDING_API_KEY}
store: {backend: milvus, uri: http://localhost:19530, collection: recipe_chunks_m3}
YAML
export EMBEDDING_API_KEY=...
```

**C. 纯冒烟（不联网、不要 Milvus）**：`--backend local --embedding hashing`，只验证链路，向量无检索语义。

> 维度一变必须换 `store.collection`，否则 Milvus 会按旧 schema 报维度不匹配。

## 5. 建库

```bash
python -m rag_core.cli build --config configs/base.yaml
INFO 已写入向量库：1701 块（后端 milvus）
建库完成：{'recipes': 322, 'chunks': 1701, ...}
```

**验收**：菜谱 322、子块约 1700、无 `template`。跑两次数字必须完全一致（索引是确定性的，这是 P0 的核心承诺）。

## 6. 检索与问答

```bash
# 只看检索：结果会打印候选块数、两路命中与耗时
python -m rag_core.cli search "简单的素菜，十分钟能做好" --category 素菜 --max-difficulty 2

# 单臂消融：同一索引上换臂
python -m rag_core.cli search "红烧肉炖多久" --arms sparse --json

# 带生成（需 LLM_API_KEY）
export LLM_API_KEY=...
python -m rag_core.cli ask "红烧肉要炖多久，怎么判断收汁好了"
```

## 7. 自检（把"跑通"变成数字）

```bash
python scripts/selfcheck.py --config configs/base.yaml
```

输出五段：索引确定性、分词消融、检索臂对比、过滤下推、上下文组装；同时写 `artifacts/selfcheck/*.json` 留档。

本地 hashing 向量的一次真实输出（仅证明机制，dense 臂数字无意义）：

```
分词               有命中   Hit@1   Hit@5     ↑ chunk 级，gold=菜名
char_bigram         100%     92%    100%
whitespace            0%      0%      0%     ← 上游 langchain 默认分词

过滤下推          平均候选  合规率
pre                   164    100%
post                 1701     40%           ← 先检索后过滤，被不合格父文档挤占
```

服务器上换真 embedding 后重跑，把 `arm` 三行与 `pre/post` 两行抄进 README 的消融表。

## 8. 常见错误

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `MilvusException: fail connecting` | 容器没起 / 端口未映射 | `docker compose ps`，`curl localhost:9091/healthz` |
| `dimension mismatch` | 换模型没换 collection | 改 `store.collection` 或 `recreate: true` |
| 检索结果全是同一篇菜谱 | `parent_strategy` 用了 `hit_count` | 改回 `rrf_sum` |
| `无候选` / 空结果 | 过滤条件与语料不符 | 先 `stats` 看类目分布，`empty_reason=no_candidate_after_filter` 就是这个 |
| `OMP: Error #15` | torch 与 numpy 的 OpenMP 冲突 | `export KMP_DUPLICATE_LIB_OK=TRUE`（仅调试） |
| `ModuleNotFoundError: jieba` | 用了 `tokenizer: jieba` | 装 `jieba`，或保持 `char_bigram` |
| 生成报 401 | 没设 `LLM_API_KEY` | 检索侧实验可忽略，`llm.model: none` 即可 |

## 9. 收尾

```bash
docker compose -f docker-compose.milvus.yml down       # 保留数据卷
docker compose -f docker-compose.milvus.yml down -v    # 连数据一起删
rm -rf artifacts/                                      # 索引与 trace 可重建
```
