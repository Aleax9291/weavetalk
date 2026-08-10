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

from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import get_graph
from app.services.llm import get_llm

router = APIRouter()
MAX_SESSIONS = 1_000

# session_id -> {"messages": [...], "profile": {...}, "clarify_rounds": int}
SESSIONS: dict[str, dict] = {}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2_000)
    session_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


def _create_session() -> tuple[str, dict]:
    """Create a bounded in-memory demo session.

    This is deliberately lightweight. Production deployments should replace it
    with authenticated, expiring server-side storage.
    """
    if len(SESSIONS) >= MAX_SESSIONS:
        SESSIONS.pop(next(iter(SESSIONS)))
    sid = uuid.uuid4().hex
    session = {"messages": [], "profile": {}, "clarify_rounds": 0}
    SESSIONS[sid] = session
    return sid, session


@router.post("/session")
async def new_session() -> dict:
    sid, _ = _create_session()
    return {"session_id": sid}


@router.post("/chat")
async def chat(req: ChatRequest) -> EventSourceResponse:
    if req.session_id is None:
        sid, session = _create_session()
    else:
        sid = req.session_id
        session = SESSIONS.get(sid)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")

    async def event_stream():
        state_in = {
            "messages": [*session["messages"], HumanMessage(content=req.message)],
            "profile": session.get("profile", {}),
            "clarify_rounds": session.get("clarify_rounds", 0),
        }
        try:
            result = await get_graph().ainvoke(state_in)
        except Exception:  # noqa: BLE001 — 顶层降级：任何异常都给用户兜底话术
            logger.exception("agent invocation failed")
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
        # 证据外发：推荐引用的评价/价格数据（评估 judge 据此核验回答数字，前端不消费）
        if result.get("evidence"):
            yield {"event": "evidence",
                   "data": json.dumps(result["evidence"], ensure_ascii=False)}
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
