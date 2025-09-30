import json
from tools import CS2MarketAgentAPI
from skin_name_manager import SkinNameManager

def build_tools(agent_api: CS2MarketAgentAPI, skin_manager: SkinNameManager) -> dict:
    """
    构建并返回一个 Agent 可以使用的工具字典。
    每个工具都包含描述、参数和要执行的函数。
    """

    def get_inventory_tool(steam_id: str) -> str:
        """工具函数：获取并格式化玩家库存"""
        print(f"--- [工具执行] get_inventory_tool, 参数: steam_id={steam_id} ---")
        inventory_data = agent_api.get_player_inventory(steam_id)
        if inventory_data:
            return agent_api.format_inventory_to_markdown(inventory_data)
        return "未能获取到库存信息。可能 ID 错误或库存为私密。"

    def get_price_overview_tool(user_query: str) -> str:
        """工具函数：匹配名称、获取并格式化实时价格"""
        print(f"--- [工具执行] get_price_overview_tool, 参数: user_query='{user_query}' ---")
        matched_name = skin_manager.find_best_match(user_query)
        if not matched_name:
            return f"无法为 '{user_query}' 找到匹配的皮肤名称。"
        
        price_data = agent_api.get_item_price_overview(matched_name, currency=23) # 默认查人民币
        if price_data:
            return agent_api.format_price_overview_to_markdown(price_data, matched_name)
        return f"为 '{matched_name}' 查询价格失败。"

    def get_price_history_tool(user_query: str) -> str:
        """工具函数：匹配名称、获取并格式化历史价格"""
        print(f"--- [工具执行] get_price_history_tool, 参数: user_query='{user_query}' ---")
        matched_name = skin_manager.find_best_match(user_query)
        if not matched_name:
            return f"无法为 '{user_query}' 找到匹配的皮肤名称。"
            
        history_data = agent_api.get_item_price_history(matched_name)
        if history_data and history_data.get('prices'):
            prices = history_data['prices']
            # 对历史数据进行总结，而不是返回整个原始数据
            summary = (
                f"成功获取 '{matched_name}' 的 {len(prices)} 条历史价格记录。\n"
                f"价格单位: {history_data.get('price_prefix', '')}\n"
                f"最早记录: {prices[0][0]} - 价格 {prices[0][1]}\n"
                f"最新记录: {prices[-1][0]} - 价格 {prices[-1][1]}"
            )
            return summary
        return f"为 '{matched_name}' 查询历史价格失败。"

    # 这是 Agent 理解工具的关键部分
    tools = {
        "get_player_inventory": {
            "description": "查询指定玩家的CS2游戏库存。需要提供玩家的64位Steam ID。",
            "args": {"steam_id": "玩家的64位Steam ID字符串"},
            "function": get_inventory_tool
        },
        "get_item_price": {
            "description": "查询某个CS2皮肤/物品的当前实时市场价格、24小时成交量和价格中位数。这是最常用的价格查询工具。",
            "args": {"user_query": "用户输入的物品名称，可以是模糊的、带别名或磨损的，例如'ak红线久经'或'awp asiimov ft'"},
            "function": get_price_overview_tool
        },
        "get_item_price_history": {
            "description": "查询某个CS2皮肤/物品的历史价格走势。当用户想了解一个物品过去的价格变化或趋势时使用。",
            "args": {"user_query": "用户输入的物品名称，可以是模糊的，例如'蝴蝶刀渐变'"},
            "function": get_price_history_tool
        }
    }
    return tools