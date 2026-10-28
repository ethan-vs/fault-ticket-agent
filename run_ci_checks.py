"""无API Key、无需下载BGE的快速回归；两项真实BGE集成留给本地完整检查。"""

import argparse
import json
import os
from pathlib import Path
import sys
import unittest

from run_quality_checks import OFFLINE_MODULES, TimedResult, summarize


LOCAL_MODEL_TESTS = {
    "test_mcp_agent.TestMCPAgent.test_real_mcp_rag_flow",
    "test_evidence_policy.TestEvidencePolicy.test_real_mcp_reports_updated_revision_and_rebuilds_index",
}


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("ci-artifacts"))
    args = parser.parse_args()
    # CI不读取用户的真实Key。测试中的模型响应均由Mock提供。
    os.environ["DEEPSEEK_API_KEY"] = "ci-placeholder-not-a-real-key"
    modules = OFFLINE_MODULES + ["test_quality_report", "test_agent_live_evaluation"]
    loaded = list(flatten(unittest.defaultTestLoader.loadTestsFromNames(modules)))
    deferred = [test.id() for test in loaded if test.id() in LOCAL_MODEL_TESTS]
    suite = unittest.TestSuite(test for test in loaded if test.id() not in LOCAL_MODEL_TESTS)
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, resultclass=TimedResult).run(suite)
    rows = list(result.records.values())
    summary = summarize(rows, [])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"scope": "ci_mock_model_and_local_protocol_checks", "summary": summary,
              "deferred_local_bge_tests": deferred, "tests": rows,
              "live_model_evaluation": False}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n快速回归：", summary)
    print("另有", len(deferred), "项真实BGE集成需在本地运行 python run_quality_checks.py 验证。")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
