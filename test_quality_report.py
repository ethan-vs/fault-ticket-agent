import io
import unittest

from run_quality_checks import TimedResult, render_html, summarize


class TestQualityReport(unittest.TestCase):
    def test_missing_modules_cannot_be_reported_as_complete(self):
        summary = summarize([{"status": "passed"}], ["test_missing"])
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["status"], "needs_review")

    def test_zero_tests_is_not_a_pass(self):
        self.assertEqual(summarize([], [])["status"], "needs_review")

    def test_failing_subtest_marks_parent_method_failed(self):
        class Example(unittest.TestCase):
            def test_example(self):
                for value in (1, 2):
                    with self.subTest(value=value):
                        self.assertEqual(value, 1)
        result = unittest.TextTestRunner(stream=io.StringIO(), resultclass=TimedResult).run(unittest.defaultTestLoader.loadTestsFromTestCase(Example))
        records = list(result.records.values())
        self.assertEqual(len(records), 1)
        self.assertEqual(summarize(records, [])["failed"], 1)

    def test_html_escapes_test_details(self):
        records = [{"id": "test.<script>alert(1)</script>", "description": "<img src=x onerror=alert(1)>", "status": "failed", "details": ["<script>bad()</script>"], "elapsed_ms": 1.0}]
        report = {
            "summary": summarize(records, []), "tests": records, "artifacts": {"latest_plan": None, "knowledge": {}},
            "packages": {}, "missing_modules": [], "elapsed_seconds": 1,
            "timestamp": "test", "project_name": "test", "manifest_sha256": "abc",
        }
        rendered = render_html(report)
        self.assertNotIn("<script>bad()", rendered)
        self.assertNotIn("<img src=x", rendered)
        self.assertIn("&lt;script&gt;", rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
