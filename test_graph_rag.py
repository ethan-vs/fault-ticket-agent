import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langgraph.checkpoint.sqlite import SqliteSaver
from openai import APITimeoutError
from openai.types.chat import ChatCompletionMessage

import graph_agent


MATCHES = [{
    "chunk_id": "RULE-SOFT-001@1.0#001",
    "document_id": "RULE-SOFT-001", "version": "1.0",
    "source": "项目自编模拟规范", "score": 0.8,
    "text": "软件启动后闪退时，应收集软件版本、崩溃日志和复现步骤。",
}]


def review_json(fake_quote=False):
    return json.dumps({
        "status": "answered",
        "selected_sources": [{
            "chunk_id": MATCHES[0]["chunk_id"],
            "evidence": "不存在于原文中的伪造摘录。" if fake_quote else "应收集软件版本、崩溃日志和复现步骤。",
            "reason": "与软件闪退现象相关。",
        }],
        "excluded_sources": [],
        "answer": "建议收集软件版本、崩溃日志和复现步骤。[RULE-SOFT-001@1.0#001]",
    }, ensure_ascii=False)


def response(content=None, tool=None, args=None, call_id="call_001"):
    calls = None if tool is None else [{
        "id": call_id, "type": "function",
        "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)},
    }]
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="tool_calls" if calls else "stop",
        message=ChatCompletionMessage(role="assistant", content=content, tool_calls=calls),
    )])


def initial_state():
    return {
        "messages": [{"role": "user", "content": "分析 TK-002，检索相关资料并引用来源。"}],
        "session_id": "rag-test", "turn_id": "turn-rag",
        "model_calls": 0, "tool_calls": 0, "status": "running",
    }


class TestGraphRAG(unittest.TestCase):
    def test_ticket_retrieval_and_cited_answer(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            response(tool="get_ticket", args={"ticket_id": "TK-002"}),
            response(tool="search_knowledge", args={"query": "网络正常但报表软件启动闪退"}, call_id="call_002"),
            response(content=review_json()),
        ]
        with patch("graph_agent.search_knowledge", return_value={"status": "ok", "matches": MATCHES}) as search, patch("graph_agent.write_event"), patch("builtins.print"):
            result = graph_agent.build_graph(client).invoke(initial_state())
        search.assert_called_once_with("网络正常但报表软件启动闪退", top_k=3)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["model_calls"], 3)
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(result["rag_review"]["status"], "answered")
        final_request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(final_request["tool_choice"], "none")
        self.assertEqual(final_request["response_format"], {"type": "json_object"})

    def test_bad_search_arguments_do_not_execute(self):
        call = {"function": {"name": "search_knowledge", "arguments": '{"query":"故障", "top_k":true}'}}
        with patch("graph_agent.search_knowledge") as search:
            result = graph_agent.dispatch_tool(call)
        search.assert_not_called()
        self.assertEqual(result["status"], "invalid_arguments")

    def test_fake_quote_fails_closed(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            response(tool="search_knowledge", args={"query": "软件闪退"}),
            response(content=review_json(fake_quote=True)),
        ]
        with patch("graph_agent.search_knowledge", return_value={"status": "ok", "matches": MATCHES}), patch("graph_agent.write_event") as log, patch("builtins.print"):
            with self.assertRaisesRegex(ValueError, "原文"):
                graph_agent.build_graph(client).invoke(initial_state())
        self.assertTrue(any(c.args[1] == "rag_validation_failed" for c in log.call_args_list))

    def test_no_documents_returns_without_generation_call(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [response(tool="search_knowledge", args={"query": "软件闪退"})]
        with patch("graph_agent.search_knowledge", return_value={"status": "no_documents", "matches": []}), patch("graph_agent.write_event"), patch("builtins.print"):
            result = graph_agent.build_graph(client).invoke(initial_state())
        self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertEqual(result["rag_review"]["status"], "insufficient_evidence")

    def test_previous_turn_retrieval_is_not_current_context(self):
        old_call = response(tool="search_knowledge", args={"query": "软件闪退"}).choices[0].message.model_dump(exclude_none=True)
        messages = [
            {"role": "user", "content": "之前的问题"}, old_call,
            {"role": "tool", "tool_call_id": "call_001", "content": json.dumps({"status": "ok", "matches": MATCHES})},
            {"role": "assistant", "content": "之前的回答"},
            {"role": "user", "content": "你好"},
        ]
        self.assertIsNone(graph_agent.current_rag_matches(messages))

    def test_resume_after_retrieval_does_not_search_again(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoints.sqlite"
            config = {"configurable": {"thread_id": "rag-resume"}}
            client = MagicMock()
            client.chat.completions.create.side_effect = [
                response(tool="search_knowledge", args={"query": "软件闪退"}),
                APITimeoutError(request=MagicMock()),
            ]
            with SqliteSaver.from_conn_string(str(path)) as saver, patch("graph_agent.search_knowledge", return_value={"status": "ok", "matches": MATCHES}), patch("graph_agent.write_event"), patch("builtins.print"):
                graph = graph_agent.build_graph(client, checkpointer=saver)
                with self.assertRaises(APITimeoutError):
                    graph.invoke(initial_state(), config=config, durability="sync")
                self.assertEqual(graph.get_state(config).next, ("model",))

            second_client = MagicMock()
            second_client.chat.completions.create.side_effect = [response(content=review_json())]
            with SqliteSaver.from_conn_string(str(path)) as saver, patch("graph_agent.search_knowledge") as search, patch("graph_agent.write_event"), patch("builtins.print"):
                graph = graph_agent.build_graph(second_client, checkpointer=saver)
                result = graph.invoke(None, config=config, durability="sync")
                self.assertEqual(graph.get_state(config).values["rag_review"]["status"], "answered")
            search.assert_not_called()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(second_client.chat.completions.create.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
