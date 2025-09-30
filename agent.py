import asyncio
import dashscope
import json
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List
from dashscope.api_entities.dashscope_response import GenerationResponse

class ReActAgent:
    REACT_PROMPT_TEMPLATE = """
今天你是专业的CS2饰品市场分析助手。请严格按照以下格式进行思考和回应，一步一步地解决用户问题。
当前时间（请在回答中保持时效性）：{current_datetime}

可用工具列表:
{tools_description}

你的任务是回答用户问题，按需使用工具。思考过程遵循以下格式：
Thought: 你当前的思考过程，分析问题，决定是否需要以及如何使用工具。
Action: 你选择的工具名称，必须是 [{tool_names}] 之一。如果你认为已经可以回答问题，则使用 'Finish' 作为工具名称。
Action Input: 一个JSON格式的字符串，包含所选工具需要的参数。如果Action是 'Finish'，则这里是给用户的最终答案。
Observation: 工具执行的结果。这个由系统填充，你不需要填写。
... (Thought/Action/Action Input/Observation 这个循环可以重复N次)

开始！

用户问题: {user_question}
{chat_history}
"""

    def __init__(self, api_key: str, model: str, tools: dict):
        dashscope.api_key = api_key
        self.model = model
        self.tools = tools
        self.max_steps = 7
        self.conversation_history: List[Dict[str, str]] = []

    def _parse_llm_output(self, llm_output: str) -> tuple[str, str, str]:
        """解析LLM的输出，提取 Thought, Action 和 Action Input"""
        thought_marker = "Thought:"
        action_marker = "Action:"
        input_marker = "Action Input:"

        thought = llm_output.split(thought_marker)[1].split(action_marker)[0].strip()
        action = llm_output.split(action_marker)[1].split(input_marker)[0].strip()
        action_input_str = llm_output.split(input_marker)[1].strip()

        return thought, action, action_input_str

    async def run(self, user_question: str) -> AsyncIterator[Dict[str, Any]]:
        """执行 ReAct Agent 循环，按步骤产出思考与执行信息"""
        history_lines = []
        for message in self.conversation_history:
            prefix = "用户" if message.get("role") == "user" else "助手"
            content = message.get("content", "")
            history_lines.append(f"{prefix}: {content}")
        chat_history = "\n".join(history_lines)

        self.conversation_history.append({"role": "user", "content": user_question})
        user_msg_index = len(self.conversation_history) - 1
        answered = False
        
        # 格式化工具描述，让LLM知道有哪些工具可用
        tools_description = "\n".join(
            f"- {name}: {info['description']} 参数: {json.dumps(info['args'])}"
            for name, info in self.tools.items()
        )
        tool_names = ", ".join(self.tools.keys())

        for step in range(self.max_steps):
            print(f"\n=========== Agent Step {step + 1} ===========")
            
            # 1. 思考 (构建Prompt并调用LLM)
            prompt = self.REACT_PROMPT_TEMPLATE.format(
                current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                tools_description=tools_description,
                tool_names=tool_names,
                user_question=user_question,
                chat_history=chat_history
            )
            
            response: GenerationResponse = await asyncio.to_thread(
                dashscope.Generation.call,
                model=self.model,
                prompt=prompt,
            )

            if response.status_code != 200:
                yield {
                    "type": "error",
                    "step": step + 1,
                    "message": f"LLM 调用失败: {response.message}",
                }
                self.conversation_history.pop(user_msg_index)
                return

            llm_output = response.output.text.strip()

            # 2. 解析LLM的决策
            try:
                thought, action, action_input_str = self._parse_llm_output(llm_output)
            except IndexError:
                yield {
                    "type": "error",
                    "step": step + 1,
                    "message": f"Agent解析输出失败，无法继续。输出为: {llm_output}",
                }
                self.conversation_history.pop(user_msg_index)
                return

            yield {
                "type": "thought",
                "step": step + 1,
                "thought": thought,
                "action": action,
                "action_input": action_input_str,
            }

            if action == "Finish":
                answered = True
                self.conversation_history.append({
                    "role": "assistant",
                    "content": action_input_str,
                })
                yield {
                    "type": "final",
                    "step": step + 1,
                    "final_answer": action_input_str,
                }
                return

            # 3. 行动 (执行工具)
            if action in self.tools:
                try:
                    action_input = json.loads(action_input_str)
                    tool_function = self.tools[action]['function']
                    observation = tool_function(**action_input)
                except Exception as e:
                    observation = f"工具执行出错: {e}"
            else:
                observation = f"错误: 尝试使用一个不存在的工具 '{action}'。"

            if isinstance(observation, str):
                observed_text = observation
            else:
                observed_text = json.dumps(observation, ensure_ascii=False)

            yield {
                "type": "observation",
                "step": step + 1,
                "observation": observed_text,
            }
            
            # 4. 观察 (将结果添加到历史记录中，准备下一次循环)
            chat_history += f"\n{llm_output}\nObservation: {observed_text}\n"

        if not answered:
            self.conversation_history.pop(user_msg_index)
            yield {
                "type": "incomplete",
                "message": "Agent 已达到最大思考步数，未能得出最终答案。",
            }

    def reset_conversation(self) -> None:
        """重置多轮对话上下文。"""
        self.conversation_history.clear()