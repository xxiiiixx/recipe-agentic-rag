# tools/（P2）

把 `rag_core` 包成 function calling 工具。每个工具只做一件事，返回**结构化证据**而不是拼好的文本
（C7 的 `tools.py` 返回 Markdown 字符串，导致反思节点只能对整段文字瞎猜）。

| 工具 | 参数 | 用途 |
|---|---|---|
| `search_recipes` | `query`, `top_k=5`, `filters?` | 主检索（双路 + RRF） |
| `filter_by_constraints` | `filters` | 只筛条件、返回候选规模，让 Agent 学会"先缩小空间再检索" |
| `get_recipe_detail` | `dish_name`, `section?` | 取整篇或指定章节（原料/操作） |
| `compare_recipes` | `dishes[]` | 多菜对比（难度、原料差异） |

## 约束

1. **只读 + 幂等**，便于重放评测；
2. 返回带 `score` 与来源臂，让反思节点能区分"召回了但分低"和"根本没召回"；
3. **空结果不是异常**：`{"hits": [], "reason": "no_candidate_after_filter"}`；
4. 参数在工具内校验，类目非法返回 `{"error": "unknown_category", "allowed": [...]}`，让模型自己纠错。

```json
{
  "name": "search_recipes",
  "description": "在菜谱库中检索相关片段，返回按融合分排序的证据列表。",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "核心检索词，保留菜名/食材/口味实体"},
      "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
      "filters": {
        "type": "object",
        "properties": {
          "category": {"type": "array", "items": {"type": "string"}, "description": "如 素菜/荤菜/汤品"},
          "difficulty_max": {"type": "integer", "minimum": 1, "maximum": 5},
          "dish_name": {"type": "array", "items": {"type": "string"}}
        }
      }
    },
    "required": ["query"]
  }
}
```
