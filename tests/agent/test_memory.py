"""Memory 系统单元测试。"""

from __future__ import annotations

import time

from src.agent.memory.conversation import ConversationBuffer
from src.agent.memory.working import WorkingMemory


class TestConversationBuffer:
    def test_add_and_get_messages(self):
        buf = ConversationBuffer(max_turns=5)
        buf.add("user", "Hello")
        buf.add("assistant", "Hi there!")

        msgs = buf.get_messages()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "Hello"}
        assert msgs[1] == {"role": "assistant", "content": "Hi there!"}

    def test_trim_excess_messages(self):
        buf = ConversationBuffer(max_turns=2)
        for i in range(6):
            buf.add("user", f"Q{i}")
            buf.add("assistant", f"A{i}")

        msgs = buf.get_messages()
        assert len(msgs) <= 4  # 2 turns * 2

    def test_ttl_expiry(self):
        buf = ConversationBuffer(ttl_seconds=0)  # 立即过期
        buf.add("user", "test")
        assert buf.is_expired()

    def test_clear(self):
        buf = ConversationBuffer()
        buf.add("user", "test")
        buf.clear()
        assert len(buf) == 0


class TestWorkingMemory:
    def test_set_and_get(self):
        wm = WorkingMemory()
        wm.set("hypothesis", "Method A is better")
        assert wm.get("hypothesis") == "Method A is better"
        assert wm.get("missing", "default") == "default"

    def test_get_all(self):
        wm = WorkingMemory()
        wm.set("a", "1")
        wm.set("b", "2")
        assert wm.get_all() == {"a": "1", "b": "2"}

    def test_clear(self):
        wm = WorkingMemory()
        wm.set("key", "value")
        wm.clear()
        assert len(wm) == 0
