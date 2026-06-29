"""Tests for the Sentinel platform. Run with: python -m pytest -q  (or unittest)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.core.config import Config
from sentinel.platform import Sentinel
from sentinel.tools.sandbox import safe_path, SandboxError
from sentinel.security.scanner import SecurityScanner


class TestSandbox(unittest.TestCase):
    def test_blocks_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SandboxError):
                safe_path(d, "../../etc/passwd")

    def test_allows_inside(self):
        with tempfile.TemporaryDirectory() as d:
            p = safe_path(d, "sub/file.txt")
            self.assertTrue(str(p).startswith(str(Path(d).resolve())))


class TestTools(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.s = Sentinel(Config(workspace=self.dir, backend="echo"))

    def test_write_and_read(self):
        wf = self.s.tools.get("write_file")
        rf = self.s.tools.get("read_file")
        self.assertIn("Wrote", wf("hello.txt|||hi there"))
        self.assertEqual(rf("hello.txt"), "hi there")

    def test_execute_python(self):
        ep = self.s.tools.get("execute_python")
        self.assertIn("42", ep("print(6*7)"))

    def test_shell_allowlist_blocks(self):
        sh = self.s.tools.get("run_shell")
        self.assertIn("Blocked", sh("rm -rf /"))

    def test_analyze_csv(self):
        Path(self.dir, "d.csv").write_text("a,b\n1,2\n3,4\n5,6\n")
        out = self.s.tools.get("analyze_data")("d.csv")
        self.assertIn("Rows: 3", out)


class TestSecurityScanner(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_detects_secret_and_eval(self):
        Path(self.dir, "bad.py").write_text(
            'API_KEY = "sk-ABCDEFGHIJKLMNOPQRSTUVWX1234567890"\n'
            'result = eval(user_input)\n'
            'import subprocess; subprocess.call(cmd, shell=True)\n'
        )
        report = SecurityScanner().scan_path(self.dir)
        rule_ids = {f.rule_id for f in report.findings}
        self.assertTrue(any(r.startswith("SECRET_") for r in rule_ids))
        self.assertIn("PY_EVAL", rule_ids)
        self.assertIn("PY_SHELL_TRUE", rule_ids)
        self.assertGreater(report.risk_score(), 0)

    def test_clean_file(self):
        Path(self.dir, "ok.py").write_text("def add(a, b):\n    return a + b\n")
        report = SecurityScanner().scan_path(self.dir)
        self.assertEqual(report.risk_score(), 0)


class TestAgent(unittest.TestCase):
    def test_agent_runs_with_echo_backend(self):
        d = tempfile.mkdtemp()
        s = Sentinel(Config(workspace=d, backend="echo", max_agent_steps=6))
        result = s.run_agent("Please calculate 12 * 11 for me")
        self.assertTrue(result["completed"])
        self.assertTrue(any(step["action"] == "execute_python"
                            for step in result["steps"]))


class TestStatus(unittest.TestCase):
    def test_no_secret_leak_in_status(self):
        s = Sentinel(Config(backend="echo"))
        st = s.status()
        self.assertNotIn("openai_api_key_env", st["config"])
        self.assertIn("openai_api_key_set", st["config"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
