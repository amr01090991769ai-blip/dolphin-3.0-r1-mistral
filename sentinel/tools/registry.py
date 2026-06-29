"""Tool registry and the built-in tool implementations.

Each tool is a callable taking a single string argument (the raw arg string
parsed from the agent's ``Action:`` line) and returning a string observation.
Multi-argument tools use ``|||`` as a separator (e.g. write_file).

All tools are *real* — no mocks — but constrained by the sandbox and config
guardrails so they are safe to expose to an autonomous agent.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import urllib.request
import urllib.error
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ..core.config import Config
from .sandbox import safe_path, ensure_workspace, SandboxError


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]
    usage: str = ""

    def __call__(self, arg: str) -> str:
        return self.func(arg)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def describe(self) -> str:
        lines = []
        for t in self._tools.values():
            lines.append(f"- {t.name}{('(' + t.usage + ')') if t.usage else ''}: {t.description}")
        return "\n".join(lines)

    def to_list(self) -> List[Dict[str, str]]:
        return [{"name": t.name, "description": t.description, "usage": t.usage}
                for t in self._tools.values()]


# --------------------------------------------------------------------------- #
# Built-in tool factory
# --------------------------------------------------------------------------- #
def build_default_registry(config: Config,
                           security_scanner=None) -> ToolRegistry:
    reg = ToolRegistry()
    ws = config.workspace
    ensure_workspace(ws)

    # ---- file tools ------------------------------------------------------ #
    def read_file(arg: str) -> str:
        try:
            p = safe_path(ws, arg.strip())
            if not p.is_file():
                return f"Error: file '{arg}' not found in workspace."
            data = p.read_text(errors="replace")
            return data if len(data) <= 8000 else data[:8000] + "\n...[truncated]"
        except SandboxError as e:
            return f"Blocked: {e}"
        except Exception as e:
            return f"Error: {e}"

    def write_file(arg: str) -> str:
        # format: path|||content
        if "|||" not in arg:
            return "Error: write_file expects 'path|||content'."
        path, content = arg.split("|||", 1)
        try:
            p = safe_path(ws, path.strip())
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return f"Wrote {len(content)} bytes to {p.relative_to(Path(ws).resolve())}."
        except SandboxError as e:
            return f"Blocked: {e}"
        except Exception as e:
            return f"Error: {e}"

    def list_files(arg: str) -> str:
        try:
            base = safe_path(ws, arg.strip() or ".")
            if not base.exists():
                return f"Error: '{arg}' not found."
            if base.is_file():
                return base.name
            entries = sorted(p.name + ("/" if p.is_dir() else "")
                             for p in base.iterdir())
            return "\n".join(entries) or "(empty)"
        except SandboxError as e:
            return f"Blocked: {e}"
        except Exception as e:
            return f"Error: {e}"

    # ---- code execution -------------------------------------------------- #
    def execute_python(arg: str) -> str:
        if not config.allow_code_exec:
            return "Code execution is disabled by configuration."
        # Run in a subprocess with a timeout, cwd pinned to the workspace.
        try:
            result = subprocess.run(
                [sys.executable, "-c", arg],
                capture_output=True, text=True,
                timeout=config.code_exec_timeout,
                cwd=ensure_workspace(ws),
            )
            out = (result.stdout or "") + (result.stderr or "")
            return out.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Error: execution exceeded {config.code_exec_timeout}s timeout."
        except Exception as e:
            return f"Error: {e}"

    def run_shell(arg: str) -> str:
        """Restricted shell: only a small allow-list of read-only commands."""
        if not config.allow_code_exec:
            return "Shell execution is disabled by configuration."
        allowed = {"ls", "cat", "head", "tail", "wc", "grep", "find",
                   "echo", "pwd", "date", "df", "du", "uname", "whoami"}
        cmd = arg.strip()
        first = cmd.split()[0] if cmd.split() else ""
        if first not in allowed:
            return (f"Blocked: '{first}' is not in the read-only allow-list "
                    f"{sorted(allowed)}.")
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=config.code_exec_timeout,
                cwd=ensure_workspace(ws),
            )
            return ((result.stdout or "") + (result.stderr or "")).strip() or "(no output)"
        except Exception as e:
            return f"Error: {e}"

    # ---- data analysis --------------------------------------------------- #
    def analyze_data(arg: str) -> str:
        """Quick stats on a CSV/JSON file in the workspace (uses stdlib)."""
        try:
            p = safe_path(ws, arg.strip())
            if not p.is_file():
                return f"Error: '{arg}' not found."
            if p.suffix.lower() == ".json":
                obj = json.loads(p.read_text())
                if isinstance(obj, list):
                    return f"JSON array with {len(obj)} items. Sample: {json.dumps(obj[:2])[:500]}"
                return f"JSON object with keys: {list(obj)[:50]}"
            # treat as CSV
            import csv, statistics
            with p.open() as f:
                rows = list(csv.reader(f))
            if not rows:
                return "Empty file."
            header, body = rows[0], rows[1:]
            summary = [f"Rows: {len(body)}  Columns: {len(header)}",
                       f"Headers: {header}"]
            # numeric column stats
            for i, col in enumerate(header):
                vals = []
                for r in body:
                    if i < len(r):
                        try:
                            vals.append(float(r[i]))
                        except ValueError:
                            pass
                if len(vals) >= 2:
                    summary.append(
                        f"  {col}: min={min(vals):.3g} max={max(vals):.3g} "
                        f"mean={statistics.mean(vals):.3g}")
            return "\n".join(summary)
        except SandboxError as e:
            return f"Blocked: {e}"
        except Exception as e:
            return f"Error: {e}"

    # ---- web tools ------------------------------------------------------- #
    def web_fetch(arg: str) -> str:
        if not config.allow_network_tools:
            return "Network tools are disabled by configuration."
        url = arg.strip()
        if not url.startswith(("http://", "https://")):
            return "Error: only http(s) URLs are allowed."
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Sentinel/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                ctype = resp.headers.get("Content-Type", "")
                raw = resp.read(200_000).decode(errors="replace")
            # strip tags crudely for text content
            import re
            if "html" in ctype:
                raw = re.sub(r"<script.*?</script>", " ", raw, flags=re.S | re.I)
                raw = re.sub(r"<style.*?</style>", " ", raw, flags=re.S | re.I)
                raw = re.sub(r"<[^>]+>", " ", raw)
                raw = re.sub(r"\s+", " ", raw).strip()
            return raw[:6000] + ("\n...[truncated]" if len(raw) > 6000 else "")
        except urllib.error.URLError as e:
            return f"Error fetching URL: {e}"
        except Exception as e:
            return f"Error: {e}"

    # ---- security -------------------------------------------------------- #
    def security_scan(arg: str) -> str:
        if security_scanner is None:
            return "Security scanner is not available."
        try:
            target = safe_path(ws, arg.strip() or ".")
            report = security_scanner.scan_path(str(target))
            return report.summary()
        except SandboxError as e:
            return f"Blocked: {e}"
        except Exception as e:
            return f"Error: {e}"

    # register them all
    reg.register(Tool("read_file", "Read a text file from the workspace.",
                      read_file, "path"))
    reg.register(Tool("write_file", "Write content to a file (path|||content).",
                      write_file, "path|||content"))
    reg.register(Tool("list_files", "List files/dirs in a workspace folder.",
                      list_files, "path"))
    reg.register(Tool("execute_python", "Run a Python snippet (sandboxed, timed).",
                      execute_python, "code"))
    reg.register(Tool("run_shell", "Run a read-only allow-listed shell command.",
                      run_shell, "command"))
    reg.register(Tool("analyze_data", "Summarise a CSV/JSON file's structure & stats.",
                      analyze_data, "path"))
    if config.allow_network_tools:
        reg.register(Tool("web_fetch", "Fetch and extract text from an http(s) URL.",
                          web_fetch, "url"))
    reg.register(Tool("security_scan", "Run a defensive security scan on a path.",
                      security_scan, "path"))
    return reg
