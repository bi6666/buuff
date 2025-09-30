import asyncio

from tools import CS2MarketAgentAPI
from skin_name_manager import SkinNameManager
from register_tools import build_tools
from agent import ReActAgent

async def main() -> None:
    # --- 安全警告：请从环境变量或安全配置文件中加载密钥 ---
    QWEN_API_KEY = "sk-c08fe5d86cd048bda352ed7aed6168cb"
    STEAM_API_KEY = "DD1ACC9B360B71D11FEC11DF6DC9DD63"

    print("--- [1/4] 初始化 API 模块 ---")
    agent_api = CS2MarketAgentAPI(api_key=STEAM_API_KEY)

    print("\n--- [2/4] 初始化皮肤名称管理器 (可能需要下载数据) ---")
    skin_manager = SkinNameManager(skin_list_filepath="skin_list.txt")

    print("\n--- [3/4] 构建并注册 MCP 工具 ---")
    tools = build_tools(agent_api, skin_manager)
    print(f"成功注册 {len(tools)} 个工具: {', '.join(tools.keys())}")

    print("\n--- [4/4] 初始化 ReAct Agent (使用 qwen-plus) ---")
    agent = ReActAgent(
        api_key=QWEN_API_KEY,
        model='qwen-plus',
        tools=tools,
    )
    print("Agent 初始化完成，准备接受任务！")
    print("=" * 40)
    tmp = 0
    while True:
        try:
            if tmp == 0:
                user_question = input("您好，我是CS2饰品助手，请问有什么可以帮您？(按 Ctrl+C 结束对话)\n> ")
            else:
                user_question = input("> ")
            final_answer = None
            async for event in agent.run(user_question):
                if event.get("type") == "thought":
                    step = event.get("step")
                    thought = event.get("thought")
                    action = event.get("action")
                    action_input = event.get("action_input")
                    print(f"[Step {step}] Thought: {thought}")
                    print(f"[Step {step}] Action: {action}")
                    print(f"[Step {step}] Action Input: {action_input}")
                elif event.get("type") == "observation":
                    print(f"[Observation] {event.get('observation')}")
                elif event.get("type") == "final":
                    final_answer = event.get("final_answer")
                elif event.get("type") == "error":
                    print(f"[错误] {event.get('message')}")
                    break
                elif event.get("type") == "incomplete":
                    print(event.get("message"))
            tmp += 1

            if final_answer is not None:
                print(f"\n[助手]: {final_answer}")
            print("-" * 40)

        except KeyboardInterrupt:
            print("\n再见！")
            agent.reset_conversation()
            break


if __name__ == '__main__':
    asyncio.run(main())