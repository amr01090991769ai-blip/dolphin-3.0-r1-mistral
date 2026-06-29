"""The Sentinel ReAct agent.

This is a *real* reasoning loop (Thought -> Action -> Observation -> ...),
not a mock. It drives an LLMEngine, parses the model's tool calls, executes
them through the ToolRegistry, and feeds observations back until the model
emits a Final Answer or the step budget is exhausted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

from .config import Config
from .llm import LLMEngine, ChatMessage
from ..tools.registry import ToolRegistry


SYSTEM_PROMPT = """You are Sentinel, a capable and responsible AI assistant.
You help with software engineering, data analysis, research, automation, and
*defensive* cybersecurity (finding and fixing vulnerabilities in code/systems
the user is authorised to work on).

Ethics & scope:
- You help users secure and improve THEIR OWN systems and authorised targets.
- You refuse to assist with unauthorised intrusion, malware for harming others,
  or bypassing security controls you do not own.
- You explain risks honestly.

You solve tasks step by step using tools. Respond using EXACTLY this protocol:

Thought: <your reasoning about what to do next>
Action: <tool_name>(<arguments>)

After each Action you will receive an Observation. Continue the loop. When you
have enough information, respond with:

Thought: <final reasoning>
Final Answer: <the complete answer for the user>

Available tools:
{tools}

Rules:
- Use ONE Action per step.
- For write_file, the argument format is: path|||content
- Keep arguments on a single line where possible.
- If no tool is needed, go straight to Final Answer.
"""


@dataclass
class AgentStep:
    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""
    final_answer: Optional[str] = None
    raw: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "final_answer": self.final_answer,
        }


class ReActAgent:
    def __init__(self, llm: LLMEngine, tools: ToolRegistry, config: Config):
        self.llm = llm
        self.tools = tools
        self.config = config

    # ----------------------------------------------------------------- #
    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(tools=self.tools.describe())

    @staticmethod
    def _parse(response: str) -> AgentStep:
        step = AgentStep(raw=response)
        # Final answer takes precedence
        fa = re.search(r"Final Answer:\s*(.*)", response, re.S)
        th = re.search(r"Thought:\s*(.*?)(?:\nAction:|\nFinal Answer:|$)",
                       response, re.S)
        if th:
            step.thought = th.group(1).strip()
        if fa:
            step.final_answer = fa.group(1).strip()
            return step
        act = re.search(r"Action:\s*([a-zA-Z_][\w]*)\s*\((.*)\)\s*$",
                        response, re.S | re.M)
        if act:
            step.action = act.group(1).strip()
            step.action_input = act.group(2).strip()
        return step

    # ----------------------------------------------------------------- #
    def run(self, goal: str,
            on_step: Optional[Callable[[AgentStep], None]] = None) -> Dict[str, Any]:
        """Execute the goal. Returns a dict with the final answer and the trace."""
        messages: List[ChatMessage] = [
            ChatMessage("system", self._system_prompt()),
            ChatMessage("user", goal),
        ]
        trace: List[AgentStep] = []

        for _ in range(self.config.max_agent_steps):
            response = self.llm.chat(messages)
            step = self._parse(response)
            trace.append(step)

            if step.final_answer is not None:
                if on_step:
                    on_step(step)
                return {
                    "goal": goal,
                    "final_answer": step.final_answer,
                    "steps": [s.to_dict() for s in trace],
                    "completed": True,
                }

            if step.action:
                tool = self.tools.get(step.action)
                if tool is None:
                    step.observation = (f"Error: unknown tool '{step.action}'. "
                                        f"Available: {self.tools.names()}")
                else:
                    step.observation = tool(step.action_input)
            else:
                step.observation = ("No valid Action or Final Answer found. "
                                    "Please follow the protocol.")

            if on_step:
                on_step(step)

            messages.append(ChatMessage("assistant", response))
            messages.append(ChatMessage("user", f"Observation: {step.observation}"))

        # step budget exhausted
        return {
            "goal": goal,
            "final_answer": "Reached the maximum number of steps without a final "
                            "answer. Partial progress is in the trace.",
            "steps": [s.to_dict() for s in trace],
            "completed": False,
        }
