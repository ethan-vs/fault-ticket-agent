"""切换工单的流程回归：模型响应被模拟，引用边界由真实图和校验器检查。"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openai.types.chat import ChatCompletionMessage

import graph_agent


SOFT = {"chunk_id": "RULE-SOFT-001@1.0#001", "text": "软件闪退时应收集软件版本和崩溃日志。"}
NET = {"chunk_id": "RULE-NET-001@1.0#001", "text": "无线无法连接时先确认影响范围和认证提示。"}


def response(content=None, tool=None, args=None, call_id="call_001"):
    calls = None if tool is None else [{
        "id": call_id, "type": "function",
        "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)},
    }]
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="tool_calls" if calls else "stop",
        message=ChatCompletionMessage(role="assistant", content=content, tool_calls=calls),
    )])


def review(chunk):
    return json.dumps({
        "status": "answered",
        "selected_sources": [{"chunk_id": chunk["chunk_id"], "evidence": [chunk["text"]], "reason": "与当前问题对应。"}],
        "excluded_sources": [],
        "answer": f"请先按规范收集信息。[{chunk['chunk_id']}]",
    }, ensure_ascii=False)


class TestTicketSwitch(unittest.TestCase):
    def run_switch(self, second_review):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            response(tool="get_ticket", args={"ticket_id": "TK-002"}, call_id="soft_ticket"),
            response(tool="search_knowledge", args={"query": "报表软件闪退"}, call_id="soft_search"),
            response(content=review(SOFT)),
            response(tool="get_ticket", args={"ticket_id": "TK-001"}, call_id="net_ticket"),
            response(tool="search_knowledge", args={"query": "无线无法连接，有线正常"}, call_id="net_search"),
            response(content=second_review),
        ]
        initial = {
            "messages": [{"role": "user", "content": "分析 TK-002"}],
            "session_id": "switch-test", "turn_id": "turn-soft",
            "model_calls": 0, "tool_calls": 0, "status": "running",
        }
        with patch("graph_agent.search_knowledge", side_effect=[
            {"status": "ok", "matches": [SOFT]},
            {"status": "ok", "matches": [NET]},
        ]) as search, patch("graph_agent.write_event"), patch("builtins.print"):
            graph = graph_agent.build_graph(client)
            first = graph.invoke(initial)
            second = graph.invoke({
                **first,
                "messages": first["messages"] + [{"role": "user", "content": "编号改为 TK-001，请重新分析"}],
                "turn_id": "turn-net", "model_calls": 0, "tool_calls": 0, "status": "running",
            })
            self.assertEqual(search.call_count, 2)
        return first, second

    def test_switch_uses_new_ticket_and_retrieval(self):
        first, second = self.run_switch(review(NET))
        self.assertEqual(first["rag_review"]["selected_sources"][0]["chunk_id"], SOFT["chunk_id"])
        self.assertEqual(second["rag_review"]["selected_sources"][0]["chunk_id"], NET["chunk_id"])
        self.assertEqual(second["model_calls"], 3)
        self.assertEqual(second["tool_calls"], 2)
        self.assertNotIn(SOFT["chunk_id"], second["messages"][-1]["content"])
        results = [m for m in second["messages"] if m["role"] == "tool"]
        ticket = json.loads(results[-2]["content"])["ticket"]
        self.assertEqual(ticket["ticket_id"], "TK-001")

    def test_old_source_cannot_be_used_as_new_retrieval_evidence(self):
        # 故意让第二轮模型仍采用旧软件片段，真实校验器必须拒绝。
        with self.assertRaises(ValueError):
            self.run_switch(review(SOFT))


if __name__ == "__main__":
    unittest.main(verbosity=2)
