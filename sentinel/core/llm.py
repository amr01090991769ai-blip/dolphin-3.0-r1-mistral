"""LLM inference engine for Sentinel.

Supports multiple backends behind one interface so the platform works
everywhere — from a laptop with no GPU to a server with a local model or
an OpenAI-compatible API:

  * "openai"       -> any OpenAI-compatible HTTP endpoint (OpenAI, vLLM, Ollama,
                      LM Studio, together, groq, ...). Selected automatically
                      when an API key + base url are available.
  * "transformers" -> HuggingFace Transformers (loads `hf_model_id`).
  * "llamacpp"     -> local GGUF file via llama-cpp-python (`gguf_path`).
  * "echo"         -> deterministic offline fallback used for tests / demos
                      so the whole platform is runnable with zero deps.

The engine exposes a single `chat(messages, **kw)` method returning text.
"""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from .config import Config


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str

    def as_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMEngine:
    def __init__(self, config: Config):
        self.config = config
        self.backend = self._resolve_backend(config.backend)
        self._client = None  # lazily initialised heavy backends

    # ------------------------------------------------------------------ #
    # Backend resolution
    # ------------------------------------------------------------------ #
    def _resolve_backend(self, requested: str) -> str:
        if requested != "auto":
            return requested
        cfg = self.config
        if cfg.openai_api_key() and cfg.openai_base_url:
            return "openai"
        if cfg.openai_api_key():  # default OpenAI cloud
            return "openai"
        if cfg.gguf_path:
            return "llamacpp"
        if cfg.hf_model_id:
            return "transformers"
        return "echo"

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.config.model,
            "hf_model_id": self.config.hf_model_id,
            "gguf_path": self.config.gguf_path,
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chat(self, messages: List[ChatMessage], **kwargs) -> str:
        msgs = [m.as_dict() if isinstance(m, ChatMessage) else m for m in messages]
        try:
            if self.backend == "openai":
                return self._chat_openai(msgs, **kwargs)
            if self.backend == "transformers":
                return self._chat_transformers(msgs, **kwargs)
            if self.backend == "llamacpp":
                return self._chat_llamacpp(msgs, **kwargs)
        except Exception as exc:  # graceful degradation
            return f"[LLM backend '{self.backend}' error: {exc}]\n" + self._chat_echo(msgs)
        return self._chat_echo(msgs)

    # ------------------------------------------------------------------ #
    # Backend: OpenAI-compatible
    # ------------------------------------------------------------------ #
    def _chat_openai(self, msgs: List[Dict[str, str]], **kwargs) -> str:
        cfg = self.config
        base = (cfg.openai_base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base}/chat/completions"
        payload: Dict[str, Any] = {
            "model": cfg.model,
            "messages": msgs,
        }
        # Some newer model families (e.g. gpt-5*) only accept default sampling
        # params and use max_completion_tokens. Adapt to avoid 400 errors.
        is_next_gen = cfg.model.startswith(("gpt-5", "o1", "o3", "o4"))
        if is_next_gen:
            payload["max_completion_tokens"] = kwargs.get("max_tokens", cfg.max_tokens)
        else:
            payload["temperature"] = kwargs.get("temperature", cfg.temperature)
            payload["top_p"] = kwargs.get("top_p", cfg.top_p)
            payload["max_tokens"] = kwargs.get("max_tokens", cfg.max_tokens)
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {cfg.openai_api_key()}")
        # Some proxies (Cloudflare-fronted) reject the default urllib UA.
        req.add_header("User-Agent", "Sentinel/1.0 (+https://sentinel.ai)")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
        return body["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------ #
    # Backend: Transformers
    # ------------------------------------------------------------------ #
    def _chat_transformers(self, msgs: List[Dict[str, str]], **kwargs) -> str:
        import torch  # type: ignore
        from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore

        if self._client is None:
            tok = AutoTokenizer.from_pretrained(self.config.hf_model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.config.hf_model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self._client = (tok, model)
        tok, model = self._client
        input_ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        out = model.generate(
            input_ids,
            max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            do_sample=True,
        )
        return tok.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)

    # ------------------------------------------------------------------ #
    # Backend: llama.cpp (GGUF)
    # ------------------------------------------------------------------ #
    def _chat_llamacpp(self, msgs: List[Dict[str, str]], **kwargs) -> str:
        from llama_cpp import Llama  # type: ignore

        if self._client is None:
            self._client = Llama(
                model_path=self.config.gguf_path,
                n_ctx=8192,
                verbose=False,
            )
        out = self._client.create_chat_completion(
            messages=msgs,
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )
        return out["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------ #
    # Backend: echo (offline deterministic fallback)
    # ------------------------------------------------------------------ #
    def _chat_echo(self, msgs: List[Dict[str, str]]) -> str:
        """A tiny rule-based stand-in so the platform is fully runnable with
        no model installed. It understands the ReAct protocol enough to drive
        simple tool calls in tests/demos and otherwise echoes a summary."""
        last_user = next((m["content"] for m in reversed(msgs)
                          if m["role"] == "user"), "")

        # If we already received a tool Observation, finish.
        if "Observation:" in last_user:
            return ("Thought: I have the result from the tool and can answer now.\n"
                    "Final Answer: Done. (echo backend — install a real model "
                    "or set OPENAI_API_KEY for full reasoning.)")

        # Heuristic: pick a tool based on keywords so demos actually do work.
        text = last_user.lower()
        if any(k in text for k in ("calculate", "compute", "math", "+", "sum", "احسب")):
            expr = re.findall(r"[-+/*.\d\s()]+", last_user)
            expr = max(expr, key=len).strip() if expr else "1+1"
            return (f"Thought: This is a calculation. I will run Python.\n"
                    f"Action: execute_python(print({expr}))")
        if any(k in text for k in ("scan", "vulnerab", "security", "audit", "افحص", "ثغر")):
            return ("Thought: A security review is requested. I'll scan the workspace.\n"
                    "Action: security_scan(.)")
        if any(k in text for k in ("write", "save", "create file", "اكتب", "احفظ")):
            return ("Thought: I should write a file.\n"
                    "Action: write_file(sentinel_note.txt|||Created by Sentinel echo backend.)")
        return ("Thought: No model backend is configured, so I will answer directly.\n"
                f"Final Answer: [echo] You said: {last_user[:400]}")
