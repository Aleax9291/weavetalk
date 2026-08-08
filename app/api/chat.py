"""对话接口：SSE 流式返回。

事件协议：
- event: intent   data: 意图路由结果 JSON（演示可观测性）
- event: token    data: 增量文本
- event: card     data: 商品卡片 JSON
- event: handoff  data: 转人工通知
- event: done     data: 结束

会话管理：内存态会话表（演示规模足够），重启即清空。
"""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import get_graph
from app.services.llm import get_llm

router = APIRouter()

# session_id -> {"messages": [...], "profile": {...}, "clarify_rounds": int}
SESSIONS: dict[str, dict] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


@router.post("/session")
async def new_session() -> dict:
    sid = uuid.uuid4().hex[:16]
    SESSIONS[sid] = {"messages": [], "profile": {}, "clarify_rounds": 0}
    return {"session_id": sid}


@router.post("/chat")
async def chat(req: ChatRequest) -> EventSourceResponse:
    sid = req.session_id or uuid.uuid4().hex[:16]
    session = SESSIONS.setdefault(
        sid, {"messages": [], "profile": {}, "clarify_rounds": 0}
    )

    async def event_stream():
        state_in = {
            "messages": [*session["messages"], HumanMessage(content=req.message)],
            "profile": session.get("profile", {}),
            "clarify_rounds": session.get("clarify_rounds", 0),
        }
        try:
            result = await get_graph().ainvoke(state_in)
        except Exception as e:  # noqa: BLE001 — 顶层降级：任何异常都给用户兜底话术
            yield {"event": "token", "data": "亲，系统有点忙，请稍后再试一下～"}
            yield {"event": "done", "data": ""}
            return

        # 可观测性：意图路由结果推给前端
        intents = result.get("intents", [])
        yield {
            "event": "intent",
            "data": json.dumps(
                [{"type": i["type"], "confidence": i.get("confidence", 0)} for i in intents],
                ensure_ascii=False,
            ),
        }

        full_text = ""
        if result.get("final_text"):
            full_text = result["final_text"]
            # 确定性回复也分块推，保持前端渲染一致
            for i in range(0, len(full_text), 8):
                yield {"event": "token", "data": full_text[i : i + 8]}
                await asyncio.sleep(0.01)
        else:
            async for delta in get_llm().chat_stream(result.get("gen_messages", [])):
                full_text += delta
                yield {"event": "token", "data": delta}
            if not full_text:
                full_text = "亲，不好意思我走神了，能再说一次吗～"
                yield {"event": "token", "data": full_text}

        for card in result.get("cards", []):
            yield {"event": "card", "data": json.dumps(card, ensure_ascii=False)}
        if result.get("handoff"):
            yield {"event": "handoff", "data": result.get("handoff_summary", "")}

        # 会话持久化（内存）
        session["messages"] = [
            *state_in["messages"], AIMessage(content=full_text)
        ]
        session["profile"] = result.get("profile", session.get("profile", {}))
        session["clarify_rounds"] = result.get("clarify_rounds", 0)

        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_stream())
