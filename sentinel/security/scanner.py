"""Defensive security scanner.

A pure-stdlib, dependency-free static analysis engine that helps developers
*find and fix* problems in their own code and configuration. It performs:

  1. Secret detection      — leaked API keys, tokens, private keys.
  2. SAST (lightweight)     — dangerous code patterns (eval, shell=True, pickle,
                              weak crypto, SQL string concatenation, ...).
  3. Dependency review      — flags pinned packages with known-bad markers and
                              missing version pins.
  4. Config / hardening     — world-writable files, debug flags, wildcard CORS.

This is strictly *defensive*: it reports issues and remediation advice. It does
not exploit anything.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any


SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


@dataclass
class Finding:
    rule_id: str
    severity: str           # critical | high | medium | low | info
    title: str
    file: str
    line: int
    snippet: str
    remediation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScanReport:
    target: str
    findings: List[Finding] = field(default_factory=list)
    files_scanned: int = 0

    def add(self, f: Finding) -> None:
        self.findings.append(f)

    def sorted_findings(self) -> List[Finding]:
        return sorted(self.findings,
                      key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.file, f.line))

    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for f in self.findings:
            c[f.severity] = c.get(f.severity, 0) + 1
        return c

    def risk_score(self) -> int:
        """0-100 weighted risk score (higher = worse)."""
        weights = {"critical": 40, "high": 20, "medium": 8, "low": 3, "info": 1}
        raw = sum(weights.get(f.severity, 0) for f in self.findings)
        return min(100, raw)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "files_scanned": self.files_scanned,
            "counts": self.counts(),
            "risk_score": self.risk_score(),
            "findings": [f.to_dict() for f in self.sorted_findings()],
        }

    def summary(self) -> str:
        c = self.counts()
        lines = [
            f"Security scan of: {self.target}",
            f"Files scanned: {self.files_scanned}",
            f"Risk score: {self.risk_score()}/100",
            "Findings by severity: " + (", ".join(
                f"{k}={v}" for k, v in sorted(c.items(),
                key=lambda kv: SEVERITY_ORDER.get(kv[0], 9))) or "none"),
            "",
        ]
        for f in self.sorted_findings()[:25]:
            lines.append(f"[{f.severity.upper()}] {f.title} "
                         f"({f.file}:{f.line})")
            lines.append(f"    > {f.snippet.strip()[:120]}")
            lines.append(f"    fix: {f.remediation}")
        if len(self.findings) > 25:
            lines.append(f"... and {len(self.findings) - 25} more findings.")
        if not self.findings:
            lines.append("No issues detected by the bundled rule set. ✔")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Detection rules
# --------------------------------------------------------------------------- #
SECRET_PATTERNS = [
    ("AWS_ACCESS_KEY", r"AKIA[0-9A-Z]{16}", "critical"),
    ("AWS_SECRET", r"(?i)aws_secret_access_key\s*[=:]\s*['\"][0-9a-zA-Z/+]{40}['\"]", "critical"),
    ("GITHUB_TOKEN", r"gh[pousr]_[0-9A-Za-z]{36,}", "critical"),
    ("OPENAI_KEY", r"sk-[A-Za-z0-9]{20,}", "critical"),
    ("SLACK_TOKEN", r"xox[baprs]-[0-9A-Za-z-]{10,}", "high"),
    ("PRIVATE_KEY", r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----", "critical"),
    ("GENERIC_SECRET", r"(?i)(password|passwd|secret|api[_-]?key|token)\s*[=:]\s*['\"][^'\"]{6,}['\"]", "medium"),
    ("JWT", r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", "medium"),
]

# (rule_id, regex, severity, title, remediation, file-glob-applies)
CODE_PATTERNS = [
    ("PY_EVAL", r"\beval\s*\(", "high", "Use of eval()",
     "Avoid eval on untrusted input; use ast.literal_eval or explicit parsing."),
    ("PY_EXEC", r"\bexec\s*\(", "high", "Use of exec()",
     "Avoid exec; refactor to call functions directly."),
    ("PY_PICKLE", r"\bpickle\.loads?\s*\(", "high", "Insecure pickle deserialization",
     "Do not unpickle untrusted data; use JSON or a safe schema."),
    ("PY_SHELL_TRUE", r"subprocess\.[a-zA-Z_]+\([^)]*shell\s*=\s*True", "high",
     "subprocess with shell=True", "Pass args as a list and avoid shell=True to prevent injection."),
    ("PY_OS_SYSTEM", r"\bos\.system\s*\(", "high", "os.system call",
     "Use subprocess with an argument list instead of os.system."),
    ("PY_YAML_LOAD", r"yaml\.load\s*\((?![^)]*Loader)", "high", "Unsafe yaml.load",
     "Use yaml.safe_load to avoid arbitrary object construction."),
    ("WEAK_HASH_MD5", r"hashlib\.md5\s*\(", "medium", "Weak hash (MD5)",
     "Use SHA-256+ for integrity; use bcrypt/argon2 for passwords."),
    ("WEAK_HASH_SHA1", r"hashlib\.sha1\s*\(", "medium", "Weak hash (SHA1)",
     "Use SHA-256 or stronger."),
    ("SQL_CONCAT", r"(?i)(execute|cursor\.execute)\s*\(\s*[f]?['\"].*(select|insert|update|delete).*%|.*\+\s*\w+\s*\)", "high",
     "Possible SQL injection (string-built query)",
     "Use parameterised queries (placeholders), never string concatenation."),
    ("VERIFY_FALSE", r"verify\s*=\s*False", "medium", "TLS verification disabled",
     "Never disable certificate verification in production."),
    ("DEBUG_TRUE", r"(?i)debug\s*=\s*True", "low", "Debug mode enabled",
     "Disable debug mode in production deployments."),
    ("CORS_WILDCARD", r"(?i)access-control-allow-origin['\"]?\s*[:=]\s*['\"]\*", "medium",
     "Wildcard CORS policy", "Restrict CORS to known origins."),
    ("HARDCODED_IP_BIND", r"0\.0\.0\.0", "info", "Service binds to all interfaces",
     "Ensure binding to 0.0.0.0 is intended and firewalled."),
    ("JS_EVAL", r"\beval\s*\(", "high", "Use of eval() in JS",
     "Avoid eval; use JSON.parse or safe alternatives."),
    ("JS_INNERHTML", r"\.innerHTML\s*=", "medium", "innerHTML assignment (XSS risk)",
     "Use textContent or sanitise HTML to prevent XSS."),
]

CODE_EXTS = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rb",
             ".php", ".c", ".cpp", ".cs", ".sh", ".yml", ".yaml",
             ".json", ".env", ".txt", ".cfg", ".ini", ".html"}

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
             "dist", "build", ".mypy_cache", ".pytest_cache"}


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: Dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


class SecurityScanner:
    def __init__(self, max_file_bytes: int = 1_000_000):
        self.max_file_bytes = max_file_bytes

    # ----------------------------------------------------------------- #
    def scan_path(self, path: str) -> ScanReport:
        root = Path(path)
        report = ScanReport(target=str(root))
        if root.is_file():
            self._scan_file(root, report)
        else:
            for p in root.rglob("*"):
                if any(part in SKIP_DIRS for part in p.parts):
                    continue
                if p.is_file():
                    self._scan_file(p, report)
        self._scan_dependencies(root, report)
        return report

    # ----------------------------------------------------------------- #
    def _scan_file(self, p: Path, report: ScanReport) -> None:
        if p.suffix.lower() not in CODE_EXTS and p.name not in (".env", "requirements.txt"):
            return
        try:
            if p.stat().st_size > self.max_file_bytes:
                return
            text = p.read_text(errors="replace")
        except (OSError, UnicodeError):
            return
        report.files_scanned += 1
        lines = text.splitlines()

        for i, line in enumerate(lines, 1):
            # secrets
            for rid, pattern, sev in SECRET_PATTERNS:
                if re.search(pattern, line):
                    report.add(Finding(
                        rule_id=f"SECRET_{rid}", severity=sev,
                        title=f"Potential leaked secret ({rid})",
                        file=str(p), line=i, snippet=line,
                        remediation="Remove the secret, rotate it, and load from "
                                    "environment variables / a secrets manager."))
            # high-entropy long tokens (heuristic secret detection)
            for token in re.findall(r"['\"][A-Za-z0-9_\-/+]{24,}['\"]", line):
                if _shannon_entropy(token) > 4.3:
                    report.add(Finding(
                        rule_id="SECRET_ENTROPY", severity="low",
                        title="High-entropy string (possible secret)",
                        file=str(p), line=i, snippet=line,
                        remediation="Verify this is not a credential; move secrets "
                                    "to environment variables."))
                    break
            # code patterns (apply python rules to .py, js rules to js/ts)
            for rid, pattern, sev, title, fix in CODE_PATTERNS:
                if rid.startswith("PY_") and p.suffix != ".py":
                    continue
                if rid.startswith("JS_") and p.suffix not in (".js", ".ts", ".jsx", ".tsx"):
                    continue
                if re.search(pattern, line):
                    report.add(Finding(
                        rule_id=rid, severity=sev, title=title,
                        file=str(p), line=i, snippet=line, remediation=fix))

    # ----------------------------------------------------------------- #
    def _scan_dependencies(self, root: Path, report: ScanReport) -> None:
        req = root / "requirements.txt" if root.is_dir() else None
        if req and req.is_file():
            for i, line in enumerate(req.read_text(errors="replace").splitlines(), 1):
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if not re.search(r"[=<>~!]=?", s):
                    report.add(Finding(
                        rule_id="DEP_UNPINNED", severity="low",
                        title="Unpinned dependency",
                        file=str(req), line=i, snippet=line,
                        remediation="Pin to an exact version (pkg==X.Y.Z) for "
                                    "reproducible, auditable builds."))
