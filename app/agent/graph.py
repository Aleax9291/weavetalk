"""LangGraph 主编排：route → execute → synthesize。

execute 节点按优先级逐意图处理（意图内部分发到导购/客服/工具子逻辑），
synthesize 把各意图产出整合为一次流式生成的消息序列。
确定性场景（转人工/攻击/兜底）直接产 final_text，跳过长文本生成。
"""

from __future__ import annotations

import json
import re

from langgraph.graph import END, START, StateGraph
from loguru import logger

from app.agent import tools
from app.agent.prompts import (
    ATTACK_REPLY,
    FALLBACK_REPLY,
    GUIDE_BUNDLE_PROMPT,
    GUIDE_CLARIFY_PROMPT,
    GUIDE_RECOMMEND_PROMPT,
    HANDOFF_REPLY,
    HANDOFF_SUMMARY_PROMPT,
    OBJECTION_PROMPT,
    PERSONA,
    QA_PROMPT,
)
from app.agent.router import route
from app.agent.state import AgentState
from app.services.llm import get_llm
from app.services.security import wrap_untrusted

MAX_CLARIFY_ROUNDS = 3

# 品类关键决策因子（与数据集 key_factors 对齐；咨询床品类的硬约束）
CATEGORY_KEY_FACTORS = {
    "四件套": ["bed_size"],
    "被芯": ["bed_size", "season"],
    "床垫": ["bed_size", "sleep_habit"],
    "枕芯": ["sleep_habit"],
    "毯类": [],
}

ENTITY_TO_PROFILE = {
    "尺寸": "bed_size", "床尺寸": "bed_size", "bed_size": "bed_size",
    "材质": "material", "material": "material",
    "季节": "season", "season": "season",
    "睡眠习惯": "sleep_habit", "sleep_habit": "sleep_habit",
    "预算": "budget", "budget": "budget",
    "场景": "purpose", "用途": "purpose", "purpose": "purpose",
}


# ---------------------------------------------------------------- 导购子逻辑
def _merge_profile(profile: dict, entities: dict) -> dict:
    merged = dict(profile)
    for k, v in entities.items():
        field = ENTITY_TO_PROFILE.get(k)
        if field and v:
            merged[field] = str(v)
    return merged


def _guess_categories(text: str) -> list[str]:
    return [c for c in CATEGORY_KEY_FACTORS if c in text] or ["四件套"]


def _missing_factors(categories: list[str], profile: dict) -> list[str]:
    missing: list[str] = []
    for cat in categories:
        for f in CATEGORY_KEY_FACTORS.get(cat, []):
            if f not in profile and f not in missing:
                missing.append(f)
    return missing


def _handle_guide(state: AgentState, intent: dict, parts: list[dict], cards: list[dict]) -> dict:
    """导购四类场景分流。返回需要更新的 state 字段。"""
    text = state["messages"][-1].content
    itype = intent["type"]
    profile = _merge_profile(state.get("profile", {}), intent.get("entities", {}))
    clarify_rounds = state.get("clarify_rounds", 0)

    if itype == "guide_vague":
        parts.append({
            "intent": itype,
            "instruction": "用户需求模糊。先给品类级建议（不要直接推具体商品），"
                           "再用 1-2 个问题锁定品类（送礼对象/使用场景/预算）。",
            "context": "（无需检索）",
        })
        return {"profile": profile}

    categories = _guess_categories(text)
    missing = _missing_factors(categories, profile)

    if itype != "guide_bundle" and missing and clarify_rounds < MAX_CLARIFY_ROUNDS:
        factor_names = {"bed_size": "床的尺寸", "season": "使用季节/空调环境",
                        "sleep_habit": "睡眠习惯（怕冷/怕热/颈椎）", "material": "材质偏好",
                        "budget": "预算"}
        parts.append({
            "intent": itype,
            "instruction": GUIDE_CLARIFY_PROMPT.replace(PERSONA, "").strip() +
                f"\n本次只问：{[factor_names.get(f, f) for f in missing][:2]}",
            "context": f"已收集的需求画像：{json.dumps(profile, ensure_ascii=False)}",
        })
        return {"profile": profile, "clarify_rounds": clarify_rounds + 1}

    # 检索 + 评价佐证（证据绑定：卡片与评价数据同源自检索结果）
    query = f"{text} {json.dumps(profile, ensure_ascii=False)}"
    found = tools.search_products(query, top_k=3)
    if not found.get("ok") or not found["items"]:
        parts.append({
            "intent": itype,
            "instruction": "检索无结果。如实告知暂时没有匹配的商品，并给出最接近的替代方向建议。",
            "context": "（无检索结果）",
        })
        return {"profile": profile, "clarify_rounds": 0}

    ids = [i["product_id"] for i in found["items"]]
    review_data = tools.analyze_reviews(ids)
    for item in found["items"]:
        cards.append({
            "product_id": item["product_id"], "name": item["name"],
            "price": item["price_range"][0], "reason": "、".join(item["tags"][:3]),
        })

    style = GUIDE_BUNDLE_PROMPT if itype == "guide_bundle" else GUIDE_RECOMMEND_PROMPT
    parts.append({
        "intent": itype,
        "instruction": style.replace(PERSONA, "").strip(),
        "context": (
            "候选商品：\n" + wrap_untrusted(json.dumps(found["items"], ensure_ascii=False, indent=1), "products")
            + "\n\n评价分析（数据可直接引用）：\n"
            + wrap_untrusted(json.dumps(review_data.get("summary", {}), ensure_ascii=False, indent=1), "reviews")
        ),
    })
    return {"profile": profile, "clarify_rounds": 0}


# ---------------------------------------------------------------- 客服 L1/L2 子逻辑
def _handle_consult(state: AgentState, intent: dict, parts: list[dict]) -> None:
    from app.services.retrieval import get_retriever

    # 检索查询以用户原话为主（保留"洗过""还能退"等关键语义），实体作补充
    # —— 仅用实体查询会丢失动作/条件语义，实测导致政策类问题召回错误条款
    text = state["messages"][-1].content
    entity_str = json.dumps(intent.get("entities", {}), ensure_ascii=False)
    query = text if text.strip() else entity_str
    results = get_retriever().search(
        query,
        kind="knowledge" if intent["type"] == "consult_policy" else None,
        top_k=3,
    )
    ctx = "\n\n".join(wrap_untrusted(r["text"], "knowledge") for r in results)
    parts.append({
        "intent": intent["type"],
        "instruction": QA_PROMPT.replace(PERSONA, "").strip(),
        "context": ctx or "（知识库未命中，如实告知需人工确认）",
    })


ORDER_ID_RE = re.compile(r"DD\d{11}")


def _handle_order(state: AgentState, parts: list[dict]) -> None:
    text = state["messages"][-1].content
    m = ORDER_ID_RE.search(text)
    if not m:
        parts.append({
            "intent": "query_order",
            "instruction": "用户想查订单/物流但没给订单号。请用户提供订单号（DD 开头），"
                           "并告知可在'我的订单'里复制。",
            "context": "（缺订单号）",
        })
        return
    result = tools.query_logistics(m.group(0))
    parts.append({
        "intent": "query_order",
        "instruction": "基于工具返回的订单/物流信息如实回答，"
                       "有最新物流节点就报最新节点，未发货就说明发货时效。",
        "context": wrap_untrusted(json.dumps(result, ensure_ascii=False), "order_data"),
    })


# ---------------------------------------------------------------- LangGraph 节点
async def execute(state: AgentState) -> dict:
    """按优先级逐意图执行，产出 answer_parts / cards / final_text。"""
    parts: list[dict] = []
    cards: list[dict] = []
    trace = list(state.get("trace", []))
    updates: dict = {"trace": trace}
    emotion = state.get("emotion", "calm")

    for intent in state.get("intents", []):
        itype = intent["type"]

        # 攻击拦截优先级最高（不能被情绪分支抢路）
        if itype == "attack":
            updates["final_text"] = ATTACK_REPLY
            trace.append({"node": "execute", "action": "attack_blocked"})
            return updates

        # L3：转人工（显式请求 / 投诉 / 支付异常 / 愤怒情绪）
        if itype in {"human_request", "complaint", "payment_issue"} or (
            emotion == "angry"
            and itype == state["intents"][0]["type"]
            and itype != "attack"
        ):
            summary = await get_llm().chat(
                [{"role": "system", "content": HANDOFF_SUMMARY_PROMPT},
                 {"role": "user", "content": state["messages"][-1].content}],
                temperature=0.2,
            )
            updates.update({
                "handoff": True,
                "handoff_reason": itype,
                "final_text": HANDOFF_REPLY,
                "handoff_summary": (summary or {}).get("content", "（摘要生成失败）"),
            })
            trace.append({"node": "execute", "action": "handoff", "reason": itype})
            return updates

        if itype in {"guide_single", "guide_multi", "guide_bundle", "guide_vague"}:
            updates.update(_handle_guide(state, intent, parts, cards))
        elif itype in {"consult_product", "consult_policy"}:
            _handle_consult(state, intent, parts)
        elif itype == "query_order":
            _handle_order(state, parts)
        elif itype == "chat":
            parts.append({
                "intent": "chat",
                "instruction": "友好简短回应闲聊（一句话），然后自然引导回家纺业务。",
                "context": "（闲聊）",
            })
        trace.append({"node": "execute", "intent": itype, "parts": len(parts)})

    if not parts:
        updates["final_text"] = FALLBACK_REPLY
    updates["answer_parts"] = parts
    updates["cards"] = cards
    return updates


async def synthesize(state: AgentState) -> dict:
    """整合各意图产出为一次生成的消息序列（端点据此流式输出）。"""
    if state.get("final_text"):
        return {}

    parts = state.get("answer_parts", [])
    need_clarify = state.get("need_clarify")
    instructions = "\n\n".join(
        f"【处理意图：{p['intent']}】\n{p['instruction']}" for p in parts
    )
    contexts = "\n\n".join(p["context"] for p in parts)

    system = PERSONA + (
        "\n\n本轮需要处理的意图与要求：\n" + instructions
        + "\n\n整合要求：多个意图在一条回复里分点答完，优先级高的在前；"
          "语气统一、过渡自然，不要出现'意图'这类内部词汇。"
    )
    if need_clarify:
        system += "\n注意：本轮识别置信度偏低，若理解可能偏差，先简短确认理解是否正确。"

    history = [
        {"role": "user" if m.type == "human" else "assistant", "content": m.content}
        for m in state["messages"][-8:]
    ]
    gen_messages = [
        {"role": "system", "content": system},
        *history[:-1],
        {"role": "user", "content": history[-1]["content"] + "\n\n" + contexts},
    ]
    trace = list(state.get("trace", []))
    trace.append({"node": "synthesize", "parts": len(parts)})
    return {"gen_messages": gen_messages, "trace": trace}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("route", route)
    g.add_node("execute", execute)
    g.add_node("synthesize", synthesize)
    g.add_edge(START, "route")
    g.add_edge("route", "execute")
    g.add_edge("execute", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
