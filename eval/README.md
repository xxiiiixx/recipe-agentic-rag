# eval/（P1）

上游 C6 有评估骨架（`BatchEvalRunner` + Faithfulness/Relevancy evaluator），但它的
`c6_response_eval_dataset.json` 里混进了生成器口水话（16 条里至少 4 条不是问题）。
拿这种集合算分，高也没意义。本目录的评估集**手写 + 抽样校准**，先做 100 条，宁少不脏。

## 评估集

以菜名为相关性单位，gold 只标到菜名，**不标 chunk**（避免 uuid 时代的脆弱锚点）。

| 类型 | 条数 | 例子 | 考什么 |
|---|---|---|---|
| 单跳事实 | 35 | 红烧肉要炖多久 | 基础召回 |
| 约束组合 | 25 | 10 分钟内能做好的简单素菜 | 过滤下推 |
| 多食材 | 20 | 有五花肉和豆瓣酱能做什么 | 稀疏路 + 实体匹配 |
| 对比/多跳 | 10 | 红烧肉和东坡肉的区别 | Agent 多步 |
| 该拒答 | 10 | 怎么做分子料理球化 | 防编造 |

留 15 条作 dev（调参用），不进报告。

## 指标

```
Recall@k   = |gold ∩ topk| / |gold|
MRR        = 1 / rank_first_relevant
nDCG@10    = 分级相关性（2=gold，1=同类同食材，0=无关）
候选存活率 = 过滤后候选块数 / 全库块数        # 解释过滤造成的召回损失
Faithfulness / Answer Relevancy              # llamaindex 口径 + 100 条人工抽检校准
Citation Precision                           # 引用的 [n] 是否落在真实来源内（generation.py 已实现）
tokens/query、p50/p95 延迟、工具调用次数      # 代价面，与收益同表
```

## 消融协议

每次只动一个变量，其余固定 `configs/base.yaml`；每个配置跑 3 次取中位数。
一个配置 = 一个 `configs/ablation/<name>.yaml`，产物落 `artifacts/<name>/`。

六组变量见 README 的消融矩阵（分词器 / 检索臂 / 过滤时机 / 父块排序 / RRF 常数 / 上下文策略）。

## 报告格式

固定四段，不许只报涨的部分：

1. 表格：配置 × 指标，bootstrap 1000 次重采样的 95% 置信区间；
2. 代价行：tokens/query、延迟、索引大小；
3. badcase：抽 10 条，按 `未召回 / 过滤过严 / 融合错 / 超预算 / 编造` 归类；
4. 结论与不确定性：样本量小的地方写明"100 条下该差异不显著"。

## badcase 归因

每条 trace（`RetrievalResult.to_trace()`）带 `filter / candidates / dense_hits / sparse_hits / fused（含两路排名）/ parents`。
先看 `candidates`：为 0 说明约束过严或解析错，不是排序问题；`dense_hits=0` 说明语义空间没覆盖；
两路都有命中但排序错，才是融合/重排的锅。这张表就是"你怎么定位 badcase"的答案。
