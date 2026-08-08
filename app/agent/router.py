"""意图路由：关键词快速通道 + LLM 多标签识别。

双通道设计（吸收 Customer-Agent 思路）：
- 确定性规则处理高优场景（转人工/攻击/辱骂），毫秒级、零成本、零幻觉
- LLM 做细粒度多意图识别（小模型降本，JSON 结构化输出）
- 按"阻断性"优先级排序输出
"""

from __future__ import annotations

import json
import re

from loguru import logger

from app.agent.prompts import ROUTER_PROMPT
from app.agent.state import AgentState, Intent
from app.services.llm import get_llm

# 阻断性优先级：数值越小越优先（业务规则，非技术规则）
PRIORITY = {
    "attack": 0, "payment_issue": 1, "complaint": 2,
    "human_request": 3, "query_order": 4,
    "consult_product": 5, "consult_policy": 5,
    "guide_single": 6, "guide_multi": 6, "guide_bundle": 6, "guide_vague": 7,
    "chat": 8,
}

# 快速通道：确定性关键词（热词可在不重启的情况下扩展为配置）
HUMAN_KEYWORDS = ["转人工", "人工客服", "找人工", "活人", "真人客服"]
ATTACK_PATTERNS = [
    r"忽略(之前|以上|所有)的?(指令|提示|规则)",
    r"ignore\s+(all\s+)?(previous|above)\s+instructions",
    r"你(就|现在)是.{0,10}(助手|模型|AI).{0,6}扮演",
    r"system\s*prompt",
]
COMPLAINT_KEYWORDS = ["投诉", "315", "曝光", "差评威胁", "消协"]

CONFIDENCE_THRESHOLD = 0.6


def _fast_path(text: str) -> list[Intent]:
    """确定性规则快速通道，命中即返回（可与其他意图共存）。"""
    hits: list[Intent] = []
    if any(k in text for k in HUMAN_KEYWORDS):
        hits.append({"type": "human_request", "confidence": 1.0, "entities": {}})
    for pat in ATTACK_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            hits.append({"type": "attack", "confidence": 1.0, "entities": {}})
            break
    if any(k in text for k in COMPLAINT_KEYWORDS):
        hits.append({"type": "complaint", "confidence": 1.0, "entities": {}})
    return hits


async def route(state: AgentState) -> dict:
    """LangGraph 节点：输出排序后的意图列表 + 情绪 + trace。"""
    last_msg = state["messages"][-1].content
    trace = list(state.get("trace", []))

    fast_hits = _fast_path(last_msg)

    history = [
        {"role": "user" if m.type == "human" else "assistant", "content": m.content}
        for m in state["messages"][-6:]
    ]
    result = await get_llm().chat(
        [{"role": "system", "content": ROUTER_PROMPT}, *history],
        model=get_llm().router_model,
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    intents: list[Intent] = []
    emotion = "calm"
    need_clarify = False
    if result and result.get("content"):
        try:
            parsed = json.loads(result["content"])
            intents = [
                i for i in parsed.get("intents", []) if i.get("type") in PRIORITY
            ]
            emotion = parsed.get("emotion", "calm")
            need_clarify = bool(parsed.get("need_clarify"))
        except json.JSONDecodeError:
            logger.warning(f"router JSON parse failed: {result['content'][:200]}")

    # 合并快速通道结果（去重，规则结果置信度最高）
    known = {i["type"] for i in intents}
    for hit in fast_hits:
        if hit["type"] not in known:
            intents.append(hit)

    # 兜底：LLM 失败或空结果时按闲聊处理（走降级话术）
    if not intents:
        intents = [{"type": "chat", "confidence": 0.3, "entities": {}}]
        need_clarify = True

    intents.sort(key=lambda i: (PRIORITY[i["type"]], -i.get("confidence", 0)))

    top = intents[0]
    if top.get("confidence", 1) < CONFIDENCE_THRESHOLD and top["type"] not in {
        h["type"] for h in fast_hits
    }:
        need_clarify = True

    trace.append({
        "node": "router",
        "intents": intents,
        "emotion": emotion,
        "need_clarify": need_clarify,
        "fast_path": [h["type"] for h in fast_hits],
    })
    logger.info(f"route: {[i['type'] for i in intents]} emotion={emotion}")

    return {
        "intents": intents,
        "need_clarify": need_clarify,
        "emotion": emotion,
        "trace": trace,
        "clarify_rounds": state.get("clarify_rounds", 0),
    }
