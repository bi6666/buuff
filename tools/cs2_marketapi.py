import requests
import time
from typing import Dict, Any, Optional
from skin_name_manager import SkinNameManager

# CS2/CS:GO 的 Steam AppID
CS2_APP_ID = 730

class CS2MarketAgentAPI:
    """
    一个用于与 Steam 市场和 CS2 库存 API交互的类 (最终修正版 v2)。
    为 AI Agent 提供获取玩家库存、饰品实时价格和历史价格的核心功能。
    """

    def __init__(self, api_key: str):
        """
        初始化 API 客户端并建立会话。
        :param api_key: 你的 Steam Web API 密钥。
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
        })
        self._initialize_session()

    def _initialize_session(self):
        """
        私有方法：访问 Steam 市场主页以获取会话 Cookie。
        """
        try:
            print("正在初始化会话并获取 Steam 社区 Cookie...")
            self.session.get("https://steamcommunity.com/market/", timeout=15)
            print("会话初始化成功。")
        except requests.exceptions.RequestException as e:
            print(f"会话初始化失败，部分市场功能可能受限: {e}")

    def get_player_inventory(self, steam_id: str) -> Optional[Dict[str, Any]]:
        """
        功能 1 : 获取指定玩家的公开 CS2 库存信息。
        添加了 Referer Header 来模拟浏览器导航，解决 400 错误。
        
        :param steam_id: 玩家的 64 位 Steam ID (必须为公开库存)。
        :return: 包含库存物品信息的字典，如果失败则返回 None。
        """
        print(f"正在查询 Steam ID 为 {steam_id} 的玩家库存...")
        url = f"https://steamcommunity.com/inventory/{steam_id}/{CS2_APP_ID}/2"
        params = {"l": "english", "count": 300}
        
        headers = {
            "Referer": f"https://steamcommunity.com/profiles/{steam_id}/inventory/"
        }
        
        try:
            # 在本次请求中临时使用添加了 Referer 的 headers
            response = self.session.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            if data and 'assets' in data and 'descriptions' in data:
                print(f"查询成功，共获取到 {len(data['assets'])} 件物品的原始数据。")
                return data
            else:
                # 这种情况也可能是一个成功的响应，但库存确实是空的
                total_inventory_count = data.get('total_inventory_count', 0)
                if total_inventory_count == 0:
                    print("查询成功，但该玩家的 CS2 库存为空。")
                else:
                    print("查询成功，但未返回有效的物品数据。")
                return data # 即使为空，也返回原始数据供上层判断

        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [403, 404]:
                print(f"查询失败: {e.response.status_code} 错误。这通常意味着对方库存是私密的或 Steam ID 无效。")
            else:
                 print(f"发生网络错误 (HTTP Error): {e}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"发生网络错误 (Request Exception): {e}")
            return None
        except ValueError:
            print("解析 JSON 响应失败。可能是被临时限制，返回了非 JSON 格式的页面。")
            return None
    def get_item_price_overview(self, market_hash_name: str, currency: int = 1) -> Optional[Dict[str, Any]]:
        """
        功能 2: 获取指定饰品的实时市场数据。
        """
        print(f"正在查询饰品 '{market_hash_name}' 的实时价格...")
        url = "https://steamcommunity.com/market/priceoverview/"
        params = {"appid": CS2_APP_ID, "currency": currency, "market_hash_name": market_hash_name}
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("success"):
                print(data)
                print("查询成功。")
                return data
            else:
                print("查询失败: 响应中 success 字段为 false。")
                return None
        except requests.exceptions.RequestException as e:
            print(f"发生网络错误: {e}")
            return None
        except ValueError:
            print("解析 JSON 响应失败。")
            return None

    def get_item_price_history(self, market_hash_name: str) -> Optional[Dict[str, Any]]:
        """
        功能 3: 获取指定饰品的历史价格数据。
        """
        print(f"正在查询饰品 '{market_hash_name}' 的历史价格...")
        url = "https://steamcommunity.com/market/pricehistory/"
        params = {"appid": CS2_APP_ID, "market_hash_name": market_hash_name}
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("success"):
                print("查询成功。")
                return data
            else:
                print("查询失败: 响应中 success 字段为 false。")
                return None
        except requests.exceptions.RequestException as e:
            print(f"发生网络错误: {e}")
            return None
        except ValueError:
            print("解析 JSON 响应失败。")
            return None

    def format_inventory_to_markdown(self, inventory_data: Dict[str, Any]) -> str:
        """
        格式化库存数据为 Markdown 表格格式。
        
        :param inventory_data: 从 get_player_inventory 获取的原始库存数据
        :return: 格式化后的 Markdown 字符串
        """
        if not inventory_data or not inventory_data.get('success'):
            return "## 库存数据格式化失败\n\n未能获取到有效的库存数据。"
        
        assets = inventory_data.get('assets', [])
        descriptions = inventory_data.get('descriptions', [])
        total_count = inventory_data.get('total_inventory_count', 0)
        
        if not assets or not descriptions:
            return f"## CS2 库存信息\n\n**总物品数量:** {total_count}\n\n库存为空或无可显示的物品。"
        
        # 创建描述信息的快速查找字典
        desc_dict = {desc['classid']: desc for desc in descriptions}
        
        markdown = f"# CS2 库存信息\n\n"
        markdown += f"**总物品数量:** {total_count}\n"
        markdown += f"**已显示物品:** {len(assets)}\n\n"
        
        markdown += "| 物品名称 | 类型 | 品质 | 稀有度 | 可交易 | 可市场交易 |\n"
        markdown += "|----------|------|------|--------|--------|------------|\n"
        
        for asset in assets:
            classid = asset.get('classid')
            desc = desc_dict.get(classid, {})
            
            # 提取基本信息
            name = desc.get('market_name', desc.get('name', '未知物品'))
            item_type = desc.get('type', '未知类型')
            
            # 提取标签信息
            tags = desc.get('tags', [])
            quality = '普通'
            rarity = '未知'
            rarity_color = ''
            
            for tag in tags:
                if tag.get('category') == 'Quality':
                    quality = tag.get('localized_tag_name', '普通')
                elif tag.get('category') == 'Rarity':
                    rarity = tag.get('localized_tag_name', '未知')
                    rarity_color = tag.get('color', '')
            
            # 格式化稀有度（如果有颜色信息）
            if rarity_color:
                rarity = f"**{rarity}**"
            
            # 交易状态
            tradable = "✅" if desc.get('tradable', 0) == 1 else "❌"
            marketable = "✅" if desc.get('marketable', 0) == 1 else "❌"
            
            markdown += f"| {name} | {item_type} | {quality} | {rarity} | {tradable} | {marketable} |\n"
        
        return markdown

    def format_price_overview_to_markdown(self, price_data: Dict[str, Any], item_name: str) -> str:
        """
        格式化价格概览数据为 Markdown 格式。
        
        :param price_data: 从 get_item_price_overview 获取的价格数据
        :param item_name: 物品名称
        :return: 格式化后的 Markdown 字符串
        """
        if not price_data or not price_data.get('success'):
            return f"## {item_name} - 价格查询失败\n\n未能获取到有效的价格数据。"
        
        markdown = f"# {item_name} - 市场价格信息\n\n"
        
        # 提取价格信息
        lowest_price = price_data.get('lowest_price', '无数据')
        median_price = price_data.get('median_price', '无数据')
        volume = price_data.get('volume', '0')
        
        markdown += "## 📊 价格概览\n\n"
        markdown += "| 指标 | 价格/数量 | 说明 |\n"
        markdown += "|------|----------|------|\n"
        markdown += f"| **最低售价** | `{lowest_price}` | 当前市场最低挂单价格 |\n"
        markdown += f"| **中位价格** | `{median_price}` | 近期交易价格中位数 |\n"
        markdown += f"| **交易量** | `{volume}` | 24小时内售出数量 |\n\n"
        
        # 添加市场活跃度评估
        try:
            volume_num = int(volume.replace(',', ''))
            if volume_num >= 100:
                activity = "🔥 高活跃度"
            elif volume_num >= 20:
                activity = "📈 中等活跃度"
            elif volume_num >= 5:
                activity = "📊 低活跃度"
            else:
                activity = "🔻 极低活跃度"
        except (ValueError, AttributeError):
            activity = "❓ 无法评估"
        
        markdown += f"**市场活跃度:** {activity}\n\n"
        
        # 添加时间戳
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        markdown += f"*数据获取时间: {current_time}*"
        
        return markdown

# --- 使用示例 ---
if __name__ == '__main__':
    MY_API_KEY = "DD1ACC9B360B71D11FEC11DF6DC9DD63" 
    QWEN_API_KEY = "sk-c08fe5d86cd048bda352ed7aed6168cb"  # 百炼 API Key
    PUBLIC_TEST_STEAM_ID = "76561198335778006"

    agent_api = CS2MarketAgentAPI(api_key=MY_API_KEY)
    skin_manager = SkinNameManager( "skin_list.txt",
                                    embedding_cache_path="skin_embeddings.npy",
                                    refresh_data=False,
                                    )


    print("\n" + "="*20 + " 1. 测试获取玩家库存 " + "="*20)
    inventory = agent_api.get_player_inventory(steam_id=PUBLIC_TEST_STEAM_ID)
    if inventory:
        print("库存数据获取成功！")
        # 格式化并打印 Markdown
        markdown_output = agent_api.format_inventory_to_markdown(inventory)
        print("\n" + "="*20 + " 格式化后的库存信息 " + "="*20)
        print(markdown_output)
    else:
        print("未能获取到库存信息。")
    
    time.sleep(2)
    item_to_query = "ak47|红线  久经"
    matched_name = skin_manager.find_best_match(item_to_query)
    print(f"匹配到的名称: {matched_name}")

    print("\n" + "="*20 + " 2. 测试获取实时价格 " + "="*20)
    price_overview = agent_api.get_item_price_overview(market_hash_name=matched_name, currency=23)
    if price_overview and price_overview.get('lowest_price'):
        print(f"完整响应: {price_overview}")
        # 格式化并打印价格信息
        price_markdown = agent_api.format_price_overview_to_markdown(price_overview, matched_name)
        print("\n" + "="*20 + " 格式化后的价格信息 " + "="*20)
        print(price_markdown)
    time.sleep(5)    
    print("\n" + "="*20 + " 3. 测试获取历史价格 " + "="*20)
    if price_history := agent_api.get_item_price_history(market_hash_name=matched_name):
        if price_history and price_history.get('prices'):
            print(f"成功获取到 {len(price_history['prices'])} 条历史价格记录。")
            print(f"最新的一条记录: {price_history['prices'][-1]}")