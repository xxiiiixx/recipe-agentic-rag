# 跑通与排错

## 1. 环境

Python 3.12，推荐 conda：

```bash
conda create -n recipe-rag python=3.12 -y
conda activate recipe-rag
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

GPU（T4）：

```bash
pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu126
pip install "sentence-transformers==6.3.0"
```

无 GPU 用 CPU 版 torch；纯冒烟（`--embedding hashing`）可以不装 torch。

## 2. Milvus

```bash
docker compose -f docker-compose.milvus.yml up -d
curl -f http://localhost:9091/healthz
```

起不来就把配置里的 `store.backend` 改成 `local`，检索结论不受影响。

## 3. 配置

```bash
cp .env.example .env   # 只填 LLM_API_KEY
```

`configs/server.yaml` 继承 `base.yaml`，只覆盖模型、维度与 collection；其余参数继承。分工见 [configs/README.md](../configs/README.md)。

模型探针（可选）：

```bash
export MODELSCOPE_API_KEY=...
python scripts/probe_models.py --models Qwen/Qwen3.5-122B-A10B
```

## 4. 建库、检索、问答

```bash
python -m rag_core.cli --config configs/server.yaml build
python -m rag_core.cli --config configs/server.yaml search "红烧肉要炖多久"
python -m rag_core.cli --config configs/server.yaml ask "红烧肉要炖多久"
python scripts/selfcheck.py --root data/C8/cook --backend local --embedding hashing --dim 256
```

## 5. 排错

| 现象 | 处理 |
|---|---|
| `fail connecting` | 容器没起；`docker compose ps`、healthz |
| `dimension mismatch` | 换模型没换 collection；改 `store.collection` 或 `recreate: true` |
| 结果全是同一篇 | `parent_strategy` 用了 `hit_count`；改回 `rrf_sum` |
| 空结果 | 过滤条件与语料不符；看 `empty_reason` |
| `OMP: Error #15` | `export KMP_DUPLICATE_LIB_OK=TRUE`（仅调试） |
| 生成 401 | 没设 `LLM_API_KEY`；检索实验可忽略 |
