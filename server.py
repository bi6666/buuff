import os
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, AsyncIterator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

from tools import CS2MarketAgentAPI
from skin_name_manager import SkinNameManager
from agent import ReActAgent


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    events: List[Dict[str, Any]]
    final_answer: str | None


agent_api: CS2MarketAgentAPI | None = None
skin_manager: SkinNameManager | None = None
react_agent: ReActAgent | None = None


def _build_tools(agent_api: CS2MarketAgentAPI, skin_manager: SkinNameManager) -> dict:
    # 与 tools.py 中一致的三个工具实现
    def get_inventory_tool(steam_id: str) -> str:
        data = agent_api.get_player_inventory(steam_id)
        if data:
            return agent_api.format_inventory_to_markdown(data)
        return "未能获取到库存信息。可能 ID 错误或库存为私密。"

    def get_price_overview_tool(user_query: str) -> str:
        matched_name = skin_manager.find_best_match(user_query)
        if not matched_name:
            return f"无法为 '{user_query}' 找到匹配的皮肤名称。"
        price_data = agent_api.get_item_price_overview(matched_name, currency=23)
        if price_data:
            return agent_api.format_price_overview_to_markdown(price_data, matched_name)
        return f"为 '{matched_name}' 查询价格失败。"

    def get_price_history_tool(user_query: str) -> str:
        matched_name = skin_manager.find_best_match(user_query)
        if not matched_name:
            return f"无法为 '{user_query}' 找到匹配的皮肤名称。"
        history_data = agent_api.get_item_price_history(matched_name)
        if history_data and history_data.get("prices"):
            prices = history_data["prices"]
            summary = (
                f"成功获取 '{matched_name}' 的 {len(prices)} 条历史价格记录。\n"
                f"价格单位: {history_data.get('price_prefix', '')}\n"
                f"最早记录: {prices[0][0]} - 价格 {prices[0][1]}\n"
                f"最新记录: {prices[-1][0]} - 价格 {prices[-1][1]}"
            )
            return summary
        return f"为 '{matched_name}' 查询历史价格失败。"

    return {
        "get_player_inventory": {
            "description": "查询指定玩家的CS2游戏库存。需要提供玩家的64位Steam ID。",
            "args": {"steam_id": "玩家的64位Steam ID字符串"},
            "function": get_inventory_tool,
        },
        "get_item_price": {
            "description": "查询某个CS2皮肤/物品的当前实时市场价格、24小时成交量和价格中位数。",
            "args": {"user_query": "用户输入的物品名称（可模糊）"},
            "function": get_price_overview_tool,
        },
        "get_item_price_history": {
            "description": "查询某个CS2皮肤/物品的历史价格走势。",
            "args": {"user_query": "用户输入的物品名称（可模糊）"},
            "function": get_price_history_tool,
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_api, skin_manager, react_agent

    qwen_api_key = os.getenv("QWEN_API_KEY", "sk-c08fe5d86cd048bda352ed7aed6168cb")
    steam_api_key = os.getenv("STEAM_API_KEY", "DD1ACC9B360B71D11FEC11DF6DC9DD63")

    agent_api = CS2MarketAgentAPI(api_key=steam_api_key)
    skin_manager = SkinNameManager(
        skin_list_filepath="skin_list.txt",
        embedding_cache_path="skin_embeddings.npy",
        name_mapping_path="skin_name_mapping.json",
        refresh_data=False,
    )

    tools = _build_tools(agent_api, skin_manager)
    react_agent = ReActAgent(api_key=qwen_api_key, model="qwen-plus", tools=tools)

    try:
        yield
    finally:
        if react_agent is not None:
            react_agent.reset_conversation()
        agent_api = None
        skin_manager = None
        react_agent = None


app = FastAPI(title="CS2 Assistant API", version="1.0.0", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    if react_agent is None:
        return AskResponse(events=[], final_answer="服务未初始化")

    events: List[Dict[str, Any]] = []
    final_answer: str | None = None

    async for event in react_agent.run(req.question):
        events.append(event)
        if event.get("type") == "final":
            final_answer = event.get("final_answer")

    return AskResponse(events=events, final_answer=final_answer)


# 便捷 GET 接口，支持浏览器直接测试：/ask?question=...
@app.get("/ask", response_model=AskResponse)
async def ask_get(question: str = Query(..., description="用户问题")) -> AskResponse:
    return await ask(AskRequest(question=question))


# Server-Sent Events 工具函数
def _sse_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chunk_text(text: str, chunk_size: int = 60) -> AsyncIterator[str]:
    async def generator() -> AsyncIterator[str]:
        for start in range(0, len(text), chunk_size):
            yield text[start : start + chunk_size]
    return generator()


@app.get("/ask/stream")
async def ask_stream(question: str = Query(..., description="用户问题")) -> StreamingResponse:
    if react_agent is None:
        async def error_stream():
            yield _sse_event({"type": "error", "message": "服务未初始化"})

        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def event_generator():
        async for event in react_agent.run(question):
            if event.get("type") == "final":
                final_answer = event.get("final_answer") or ""
                if final_answer:
                    async for piece in _chunk_text(final_answer):
                        yield _sse_event({"type": "delta", "content": piece})
                yield _sse_event({
                    "type": "final",
                    "final_answer": final_answer,
                    "step": event.get("step"),
                })
            else:
                yield _sse_event(event)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# 新对话重置接口
@app.post("/reset")
async def reset_conversation() -> Dict[str, str]:
    if react_agent is not None:
        react_agent.reset_conversation()
    return {"status": "reset"}


# 将 frontend 目录挂载为静态站点，使得访问 / 即可打开前端
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)


