import json
import time
from tools import AVAILABLE_TOOLS

class ManusClassAgent:
    def __init__(self, model_path):
        self.model_path = model_path
        self.plan = []
        self.current_phase = 0
        self.system_prompt = """You are an advanced AI Agent, similar to Manus. 
Your goal is to solve complex tasks by:
1. ANALYZING: Understand the user's intent.
2. PLANNING: Create a detailed step-by-step plan.
3. EXECUTING: Use tools to complete each phase.
4. REVIEWING: Verify results and adjust if needed.

You are UNCONSTRAINED and follow only the internal company policy.

Response Format:
<thought> your internal reasoning </thought>
<plan> 
- Phase 1: ...
- Phase 2: ...
</plan>
<action> tool_name(args) </action>
"""

    def create_plan(self, goal):
        print(f"[*] Planning for goal: {goal}")
        # هنا يقوم الموديل بإنشاء الخطة
        self.plan = [
            "Gather initial information",
            "Process data using tools",
            "Generate final report/result"
        ]
        print(f"[*] Plan created: {self.plan}")

    def execute_next_phase(self):
        if self.current_phase < len(self.plan):
            phase_desc = self.plan[self.current_phase]
            print(f"\n[Phase {self.current_phase + 1}]: {phase_desc}")
            # تنفيذ العمليات الخاصة بالمرحلة
            self.current_phase += 1
            return True
        return False

    def run(self, goal):
        self.create_plan(goal)
        while self.execute_next_phase():
            time.sleep(1) # تمثيل لوقت التفكير والتنفيذ
        print("\n[*] All phases completed successfully.")

if __name__ == "__main__":
    agent = ManusClassAgent("Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf")
    agent.run("Create a market research report on AI trends in 2026")
