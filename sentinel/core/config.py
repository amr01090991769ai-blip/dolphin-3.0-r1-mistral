"""Configuration loading for Sentinel.

Configuration is resolved with the following precedence (highest first):
  1. Environment variables (SENTINEL_*)
  2. A config file (config.json / sentinel.json) if present
  3. Sensible built-in defaults

No secrets are ever hard-coded; API keys come exclusively from the environment.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Config:
    # --- LLM backend selection ---------------------------------------------
    # backend: one of "auto", "openai", "transformers", "llamacpp", "echo"
    backend: str = "auto"

    # OpenAI-compatible endpoint (works with OpenAI, local vLLM, Ollama, etc.)
    openai_base_url: Optional[str] = None
    openai_api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4o-mini"

    # Local model paths (Transformers id or local GGUF file)
    hf_model_id: Optional[str] = None
    gguf_path: Optional[str] = None

    # --- generation defaults ----------------------------------------------
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 2048

    # --- agent settings ----------------------------------------------------
    max_agent_steps: int = 12
    # Workspace that tools are restricted to (path traversal is blocked).
    workspace: str = field(default_factory=lambda: os.environ.get(
        "SENTINEL_WORKSPACE", str(Path.cwd() / "workspace")))

    # --- security guardrails ----------------------------------------------
    allow_code_exec: bool = True
    code_exec_timeout: int = 15
    allow_network_tools: bool = True

    def openai_api_key(self) -> Optional[str]:
        # Resolve from the environment only (never persisted). When the user
        # explicitly set a custom env var, honour it first; otherwise prefer
        # the GenSpark proxy key, then standard fallbacks.
        order = []
        if self.openai_api_key_env != "OPENAI_API_KEY":
            order.append(self.openai_api_key_env)
        order += ["GSK_API_KEY", "OPENAI_API_KEY", "GENSPARK_TOKEN", "GSK_TOKEN"]
        for var in order:
            val = os.environ.get(var)
            if val:
                return val
        return None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # never leak secrets
        d.pop("openai_api_key_env", None)
        d["openai_api_key_set"] = bool(self.openai_api_key())
        return d


def _coerce(value: str) -> Any:
    """Coerce an env-string into bool/int/float when it looks like one."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _autodetect_openai_base_url(data: Dict[str, Any]) -> None:
    """Pick up an OpenAI-compatible endpoint from the environment or a
    GenSpark-style ~/.genspark_llm.yaml file (base_url only — never the key).
    The API key is always resolved at call time from the environment."""
    if data.get("openai_base_url"):
        return
    env_base = os.environ.get("OPENAI_BASE_URL")
    if env_base:
        data["openai_base_url"] = env_base
        return
    cfg = Path.home() / ".genspark_llm.yaml"
    if cfg.is_file():
        try:
            for line in cfg.read_text().splitlines():
                line = line.strip()
                if line.startswith("base_url:"):
                    data.setdefault("openai_base_url",
                                    line.split(":", 1)[1].strip())
        except OSError:
            pass


def load_config(path: Optional[str] = None) -> Config:
    data: Dict[str, Any] = {}

    # 1. config file
    candidates = []
    if path:
        candidates.append(path)
    candidates += ["sentinel.json", "config.json"]
    for c in candidates:
        p = Path(c)
        if p.is_file():
            try:
                raw = json.loads(p.read_text())
                # only pick keys that Config understands
                valid = {f for f in Config.__dataclass_fields__}
                data.update({k: v for k, v in raw.items() if k in valid})
                break
            except (json.JSONDecodeError, OSError):
                continue

    # 2. environment overrides (SENTINEL_<UPPER_FIELD>)
    for fname in Config.__dataclass_fields__:
        env_key = f"SENTINEL_{fname.upper()}"
        if env_key in os.environ:
            data[fname] = _coerce(os.environ[env_key])

    # 3. auto-detect an OpenAI-compatible endpoint (base url only)
    _autodetect_openai_base_url(data)

    return Config(**data)
