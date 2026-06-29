"""High-level Sentinel facade that wires everything together."""
from __future__ import annotations

from typing import Any, Dict, Optional

from .core.config import Config, load_config
from .core.llm import LLMEngine, ChatMessage
from .core.agent import ReActAgent
from .tools.registry import build_default_registry
from .security.scanner import SecurityScanner


class Sentinel:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()
        self.llm = LLMEngine(self.config)
        self.scanner = SecurityScanner()
        self.tools = build_default_registry(self.config, security_scanner=self.scanner)
        self.agent = ReActAgent(self.llm, self.tools, self.config)

    # --- chat (single turn, no tools) ---------------------------------- #
    def chat(self, prompt: str, system: Optional[str] = None) -> str:
        msgs = []
        if system:
            msgs.append(ChatMessage("system", system))
        msgs.append(ChatMessage("user", prompt))
        return self.llm.chat(msgs)

    # --- agent (multi-step, tool-using) -------------------------------- #
    def run_agent(self, goal: str, on_step=None) -> Dict[str, Any]:
        return self.agent.run(goal, on_step=on_step)

    # --- security ------------------------------------------------------ #
    def scan(self, path: str) -> Dict[str, Any]:
        return self.scanner.scan_path(path).to_dict()

    # --- introspection ------------------------------------------------- #
    def status(self) -> Dict[str, Any]:
        return {
            "version": __import__("sentinel").__version__,
            "llm": self.llm.info,
            "tools": self.tools.to_list(),
            "config": self.config.to_dict(),
        }
