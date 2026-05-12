import json
import time
from tools import AVAILABLE_TOOLS

class DolphinAgent:
    def __init__(self, model_path):
        self.model_path = model_path
        self.system_prompt = """You are Dolphin, an autonomous AI agent. 
You can use tools to achieve goals. 
Available tools: {tools_list}

Response format:
Thought: <your reasoning>
Action: <tool_name>(<arguments>)
Observation: <result of the action>
... (repeat until finished)
Final Answer: <your final response to the user>
""".format(tools_list=list(AVAILABLE_TOOLS.keys()))

    def run(self, user_goal):
        print(f"[*] Goal received: {user_goal}")
        history = [{"role": "system", "content": self.system_prompt}, 
                   {"role": "user", "content": user_goal}]
        
        for i in range(10): # الحد الأقصى للخطوات
            # هنا نقوم باستدعاء الموديل (تمثيل بسيط للعملية)
            # في الواقع سنستخدم llama-cpp-python أو transformers هنا
            response = self.mock_model_call(history) 
            print(f"\n[Dolphin]: {response}")
            
            if "Final Answer:" in response:
                break
            
            # استخراج الأداة والبارامترات (بسيط لأغراض العرض)
            if "Action:" in response:
                action_line = [line for line in response.split('\n') if "Action:" in line][0]
                tool_call = action_line.replace("Action:", "").strip()
                # تنفيذ الأداة
                tool_name = tool_call.split('(')[0]
                args = tool_call.split('(')[1].replace(')', '')
                
                if tool_name in AVAILABLE_TOOLS:
                    print(f"[*] Executing tool: {tool_name} with args: {args}")
                    observation = AVAILABLE_TOOLS[tool_name](args)
                    print(f"[*] Observation: {observation}")
                    history.append({"role": "assistant", "content": response})
                    history.append({"role": "user", "content": f"Observation: {observation}"})
                else:
                    print(f"[!] Tool {tool_name} not found.")
                    break
        
        print("\n[*] Task completed.")

    def mock_model_call(self, history):
        # هذا تمثيل لكيفية تفكير الموديل
        last_message = history[-1]["content"]
        if "Observation:" in last_message:
            return "Thought: I have the information now.\nFinal Answer: Task finished successfully."
        return "Thought: I need to write a script to check system status.\nAction: write_file('status.txt', 'System is OK')\nObservation: pending"

if __name__ == "__main__":
    agent = DolphinAgent("Dolphin3.0-Llama3.1-8B-Q4_K_M.gguf")
    agent.run("Check my system status and save it to a file called status.txt")
