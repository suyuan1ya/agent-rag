"""工作记忆 — 单轮内 Agent 步骤间持久化的 key-value scratchpad。

Agent 在一个推理轮次中可能需要多步（检索 → 评估 → 改写 → 再检索 → 综合），
WorkingMemory 让 Agent 在所有步骤间共享中间发现、假设和部分结论。
"""

from __future__ import annotations


class WorkingMemory:
    """Agent 单轮推理的工作记忆。

    使用场景:
    - "用户问的是方法A和方法B的比较" （记录问题意图）
    - "第一轮检索已找到方法A的定义在第3节" （记录阶段性发现）
    - "当前假设: 方法A比方法B更适用于小数据集" （记录待验证假设）
    """

    def __init__(self):
        self._store: dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        self._store[key] = value

    def get(self, key: str, default: str = "") -> str:
        return self._store.get(key, default)

    def get_all(self) -> dict[str, str]:
        return dict(self._store)

    def clear(self) -> None:
        self._store.clear()

    def to_prompt_context(self) -> str:
        """将工作记忆格式化为可注入 prompt 的文本。"""
        if not self._store:
            return ""
        lines = ["[工作记忆]"]
        for k, v in self._store.items():
            lines.append(f"  {k}: {v[:300]}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store
