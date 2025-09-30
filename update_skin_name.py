import json
import requests

# 【已更新】定义新的数据源 URL 和输出文件的名称
SOURCE_URL = "https://raw.githubusercontent.com/ByMykel/CSGO-API/main/public/api/en/all.json"
TARGET_URL = "https://raw.githubusercontent.com/ByMykel/CSGO-API/main/public/api/zh-CN/all.json"
OUTPUT_FILE = "skin_list.txt"
MAPPING_FILE = "skin_name_mapping.json"

def update_skin_list():
    """
    (v3) 从 ByMykel 的 CSGO-API 的 all.json（英文与简体中文版本）获取最新的物品数据，
    解析字典结构，提取所有有效的市场名称 (market_hash_name 或 name)，
    并将去重后的结果保存到 skin_list.txt 文件中。
    """
    source_map = {
        "en": SOURCE_URL,
        "zh-CN": TARGET_URL,
    }

    all_skin_names = []
    alias_map = {}
    english_primary_by_id = {}

    for locale, url in source_map.items():
        print(f"[*] 正在下载 {locale} 版本的物品数据...")
        print(f"    URL: {url}")

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            print(f"[*] {locale} 数据下载成功，正在解析 JSON...")
            all_items_dict = response.json()

            print("[*] 正在筛选有效的市场名称 (market_hash_name 或 name)...")

            for item_id, item in all_items_dict.items():
                candidate_raw = [
                    item.get("market_hash_name"),
                    item.get("market_name"),
                    item.get("name"),
                ]

                candidates = []
                seen = set()
                for name in candidate_raw:
                    if not name:
                        continue
                    normalized = str(name).strip()
                    if normalized and normalized not in seen:
                        candidates.append(normalized)
                        seen.add(normalized)

                if not candidates:
                    continue

                all_skin_names.extend(candidates)

                primary_en = candidates[0]

                if locale == "en":
                    english_primary_by_id[item_id] = primary_en
                    base_name = primary_en
                else:
                    base_name = english_primary_by_id.get(item_id, primary_en)

                for alias in candidates:
                    alias_map[alias] = base_name

        except requests.exceptions.RequestException as e:
            print(f"\n[X] 错误: 下载 {locale} 数据时发生网络错误。")
            print(f"    详细信息: {e}")
        except ValueError:
            print(f"\n[X] 错误: 解析 {locale} 版本 JSON 数据失败。源文件可能已损坏或格式不正确。")
        except Exception as e:
            print(f"\n[X] 下载 {locale} 版本时发生未知错误: {e}")

    unique_skin_names = list(dict.fromkeys(all_skin_names))

    if not unique_skin_names:
        print("[!] 警告: 未能在下载的数据中找到任何有效的皮肤名称。")
        return

    print(f"[*] 筛选完成，共找到 {len(unique_skin_names)} 个唯一名称。")
    print(f"[*] 正在将名称写入文件: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(unique_skin_names))

    print(f"[*] 正在写出名称映射文件: {MAPPING_FILE}")
    mapping_payload = {"alias_to_en": alias_map}
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping_payload, f, ensure_ascii=False, indent=2)

    print(f"\n[✔] 操作成功！")
    print(f"[✔] 一个包含 {len(unique_skin_names)} 个物品的超完整列表已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    update_skin_list()