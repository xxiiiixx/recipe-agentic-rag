"""P2 Agent 层：工具、查询理解、状态机。

对外只暴露三样东西：``ToolBox``（工具）、``build_understander``（查询理解）、
``RecipeAgent``（状态机）。其余都是内部实现，换编排框架时不需要动它们。
"""

from .graph import AgentRun, AgentState, Evidence, RecipeAgent, Reflection
from .tools import ToolBox, ToolResult, ToolSpec, recipe_profile
from .understand import ParsedQuery, RuleUnderstander, build_understander

__all__ = [
    "AgentRun",
    "AgentState",
    "Evidence",
    "ParsedQuery",
    "RecipeAgent",
    "Reflection",
    "RuleUnderstander",
    "ToolBox",
    "ToolResult",
    "ToolSpec",
    "build_understander",
    "recipe_profile",
]
