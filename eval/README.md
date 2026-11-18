# 评测集

P1 的评测集：105 条 test + 15 条 dev。相关性单位是**菜名**（`parent_id`），不是 chunk。

标到菜名而不是 chunk，是因为 `chunk_id` 会随切块配置变化；菜名在语料不变时稳定。同一份 qrels 才能同时评分词、过滤、父块策略和 Agent，不会切块一改就全部失效。

## 类型与数量

| type | test | dev | 例子 | 考什么 | gold |
|---|---:|---:|---|---|---|
| `single_hop` | 35 | 5 | 红烧肉要炖多久？ | 基础召回 | 同名菜的全部变体 |
| `constraint` | 25 | 4 | 10 分钟能做好的简单素菜 | 过滤下推 | 3–5 道代表菜 + 合规率 |
| `multi_ingredient` | 20 | 3 | 有鸡蛋和西红柿能做什么？ | 稀疏 + 实体匹配 | 3–5 道代表菜 |
| `compare` | 8 | 1 | 糖醋排骨和糖醋里脊有什么区别？ | 多菜同时召回 | 涉及的每道菜 |
| `multi_hop` | 7 | 1 | 推荐一道简单荤菜和一道凉拌素菜 | Agent 多步 | 每个子目标的代表菜 |
| `refusal` | 10 | 1 | 怎么做分子料理球化？ | 防编造 | 空 |

dev 不进报告，只用于调参和 prompt 调整。test 与 dev 按类型分层抽取。

## 记录格式

```json
{"qid":"s001","split":"test","type":"single_hop","query":"红烧肉要炖多久？","filters":{},"gold_parent_ids":["e11778334a7bb95b9adf98145d0f888c"],"gold_dishes":["简易红烧肉"],"expected_reason":null,"notes":"同名菜变体都算对"}
```

| 字段 | 说明 |
|---|---|
| `qid` | 唯一，类型前缀 + 序号 |
| `type` | 上表六种 |
| `filters` | 与 `SearchFilter` 同构；只有 `constraint` 必填 |
| `gold_parent_ids` | 稳定主键；`refusal` 为空 |
| `gold_dishes` | 人工可读，必须与 `parent_ids` 一一对应 |
| `expected_reason` | `refusal` 必填，写清缺什么 |
| `notes` | 标注理由、限制或歧义说明 |

## gold 规则

1. **单跳**：同名菜的全部变体都进 gold。红烧肉有 4 个变体、陈皮排骨汤有 2 份，只选其中一个会造成假阴性。
2. **约束**：语料没有烹饪时长，`10 分钟` 用 `difficulty_max=2` 代理，报告里必须写明。gold 选 3–5 道最符合意图的菜，同时单独统计 top-k 的约束合规率：合规率高但 gold 命中低，说明召回没覆盖“好答案”，不是排序问题。
3. **多食材**：gold 选 3–5 道原料覆盖查询食材的菜；另统计 top-k 的食材覆盖率，只作辅助。
4. **对比 / 多跳**：涉及的每道菜都进 gold；Recall 要求全中，MRR 看第一个命中的排名。
5. **拒答**：gold 为空，`expected_reason` 写清缺什么；答案只要给出具体菜谱或步骤即判错。

## 指标

- Recall@5/10、MRR、nDCG@10（P1 用二值相关性，另取 30 条人工分级校准）
- 候选存活率、约束合规率、食材覆盖率、拒答准确率
- Citation Precision（规则）、Faithfulness / Answer Relevancy（LLM judge + 20% 人工抽检）
- tokens/query、p50/p95、工具调用次数

检索指标为确定性单次 + 1000 次 query bootstrap 95% CI；Agent 指标为 3 次中位数。

## 标注流程

1. LLM 从菜谱正文生成候选 query；`single_hop` 保留菜名，其他类型不得出现目标菜名。
2. 人工逐条审：改写成中文口语、定 `type` / `filters` / gold、删除重复与歧义。
3. 用 `scripts/validate_eval.py` 校验 schema、`parent_id` 存在、gold 非空、`qid` 唯一、数量配额。
4. 冻结 v1；dev/test 按类型分层。评测集改动必须升版本，旧结果不覆盖。

## 已知限制

- 约束类只把 `category` / `difficulty` 编码进 `SearchFilter`；减脂、带饭、一人食、待客等语义约束无法机器校验，靠人工 gold 体现，合规率只检查结构化字段。
- `multi_ingredient` 的 gold 取“全命中食材”的菜；语料中不足 3 道时取全部。
- 约束类用难度代理时长，语料没有真实烹饪时长。
- nDCG 的“同类同食材”分级只在 30 条子集做，其余用二值。
- 同名菜变体答案可能不同（如陈皮排骨汤两份），gold 全收，不评价“选错变体”。
