"""LangGraph 主编排：route → execute → synthesize。

execute 节点按优先级逐意图处理（意图内部分发到导购/客服/工具子逻辑），
synthesize 把各意图产出整合为一次流式生成的消息序列。
确定性场景（转人工/攻击/兜底）直接产 final_text，跳过长文本生成。
"""

from __future__ import annotations

import json
import re

from langgraph.graph import END, START, StateGraph

from app.agent import tools
from app.agent.prompts import (
    ATTACK_REPLY,
    FALLBACK_REPLY,
    GUIDE_BUNDLE_PROMPT,
    GUIDE_CLARIFY_PROMPT,
    GUIDE_RECOMMEND_PROMPT,
    HANDOFF_REPLY,
    HANDOFF_SUMMARY_PROMPT,
    PERSONA,
    QA_PROMPT,
)
from app.agent.router import route
from app.agent.state import AgentState
from app.services.llm import get_llm
from app.services.security import wrap_untrusted

MAX_CLARIFY_ROUNDS = 3

# 品类关键决策因子（与数据集 key_factors 对齐；咨询床品类的硬约束）
# 床垫只要床尺寸：店内床垫是软硬两用双面设计，不依赖睡眠习惯分流（v3 GS-08 教训）
CATEGORY_KEY_FACTORS = {
    "四件套": ["bed_size"],
    "被芯": ["bed_size", "season"],
    "床垫": ["bed_size"],
    "枕芯": ["sleep_habit"],
    "毯类": [],
}

# 品类别名 → 标准品类（v1 教训："蚕丝被""枕头"不含"被芯""枕芯"字样，
# 关键词匹配落空后错误回退到"四件套"，导致问床尺寸等错位澄清——GS-03/05/06）
CATEGORY_ALIASES = {
    "四件套": ["四件套", "床品套件", "套件"],
    "被芯": ["被芯", "被子", "蚕丝被", "羽绒被", "鹅绒被", "大豆纤维被", "夏被", "冬被", "被芯"],
    "床垫": ["床垫"],
    "枕芯": ["枕芯", "枕头", "乳胶枕", "羽绒枕"],
    "毯类": ["毯类", "盖毯", "毛毯", "空调毯", "午休", "毯"],
}

# 睡眠习惯关键词 → 需求画像（用户已自述时不应再澄清——GS-06"颈椎不好"）
SLEEP_HABIT_HINTS = {
    "颈椎": "颈椎需要支撑", "脖子": "颈椎需要支撑",
    "怕热": "怕热", "怕冷": "怕冷", "失眠": "助眠需求",
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
    """按别名识别品类；一个都识别不出时返回 []（交给全库检索，不强行假设品类）。"""
    cats = [c for c, aliases in CATEGORY_ALIASES.items() if any(a in text for a in aliases)]
    return cats


_CATALOG_CACHE: str | None = None


def _catalog_overview() -> str:
    """静态品类目录（品类 → 商品名），供模糊导购/澄清锚定，防"店里没有X"幻觉。"""
    global _CATALOG_CACHE
    if _CATALOG_CACHE is None:
        products = tools._load("products.json")
        cats: dict[str, list[str]] = {}
        for p in products:
            base = p["name"].split(" ")[0]
            names = cats.setdefault(p["category"], [])
            if base not in names:
                names.append(base)
        _CATALOG_CACHE = "；".join(f"{c}（{'、'.join(names)}）" for c, names in cats.items())
    return _CATALOG_CACHE


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
    # 确定性实体兜底：路由实体抽取不稳定（v2 GS-01"1.8米的床"未入画像导致误澄清），
    # 床尺寸/睡眠习惯这类高确定性信息用规则直接兜底
    if not profile.get("bed_size"):
        m_size = re.search(r"(\d\.\d)\s*(?:米|m|M)", text)
        if m_size:
            profile["bed_size"] = f"{m_size.group(1)}米"
    # 用户在原话里自述的睡眠习惯直接入画像，不再重复澄清
    for kw, habit in SLEEP_HABIT_HINTS.items():
        if kw in text:
            profile.setdefault("sleep_habit", habit)
    clarify_rounds = state.get("clarify_rounds", 0)

    # 画像床尺寸 → 商品规格（"2.0米"→"2.0m床"），检索时优先匹配规格。
    # 终验 H-GS-05 教训：不传规格时同名去重可能留下错误床型的款
    psize = None
    if profile.get("bed_size"):
        m_ps = re.search(r"(\d\.\d)", str(profile["bed_size"]))
        if m_ps:
            psize = f"{m_ps.group(1)}m床"

    if itype == "guide_vague":
        # 静态全品类目录锚定：模糊需求的品类级建议必须基于店内真实在售结构。
        # v2 教训：按 query 检索的锚定在有干扰词时会漏品类，模型遂谎称"没有毯子"（GV-06）
        parts.append({
            "intent": itype,
            "instruction": "用户需求模糊。先给品类级建议（基于店内真实品类结构，"
                           "不要说店里没有某个在售品类；不要直接推具体商品、不要报价格）；"
                           "用户纠结两个品类时，先帮其区分使用场景（季节/空调房/用途）再给方向；"
                           "然后澄清锁定需求：想要什么品类、床的尺寸、预算，三项都用自然的话带到"
                           "（送礼场景则为：送礼对象、预算）。",
            "context": f"店内品类结构：{_catalog_overview()}",
        })
        return {"profile": profile}

    categories = _guess_categories(text)

    # 多类/套装：用户点名了要多个品类，直接按品类逐个检索推荐，不做单因子澄清
    # （v1 教训：GM-01/02 因澄清逻辑或单路检索漏品类被判"漏答"）
    if itype in {"guide_multi", "guide_bundle"}:
        if itype == "guide_bundle":
            # 套装默认范围：四件套+被芯+枕芯；文本点名的品类补充进来（如明确提到床垫）
            categories = list(dict.fromkeys(["四件套", "被芯", "枕芯"] + categories))
        elif categories:
            pass  # 用识别到的品类
        else:
            categories = []  # 全库检索
        items = []
        if categories:
            for cat in categories:
                # 套装每品类取 3 款：预算约束下需要低价备选（如大豆纤维被），
                # 只取 top2 会把性价比款挡在候选外（v2 冒烟 GM-04 超预算的教训）
                per_cat = 3 if itype == "guide_bundle" else 2
                r = tools.search_products(f"{text} {cat}", top_k=per_cat, category=cat,
                                          size=psize)
                if r.get("ok"):
                    items.extend(r["items"])
        else:
            r = tools.search_products(f"{text} {json.dumps(profile, ensure_ascii=False)}", top_k=4)
            if r.get("ok"):
                items.extend(r["items"])
        budget_ctx = ""
        if itype == "guide_bundle" and profile.get("budget"):
            budget_ctx = f"\n\n用户硬预算：{profile['budget']} 元——组合总价必须 ≤ 此预算，超出就降级替换。"
        return _recommend_parts(state, itype, items, parts, cards, profile,
                                extra_context=budget_ctx + (_promo_context() if itype == "guide_bundle" else ""))

    missing = _missing_factors(categories, profile) if categories else []

    if categories and missing and clarify_rounds < MAX_CLARIFY_ROUNDS:
        factor_names = {"bed_size": "床的尺寸", "season": "使用季节/空调环境",
                        "sleep_habit": "睡眠习惯（怕冷/怕热/颈椎）", "material": "材质偏好",
                        "budget": "预算"}
        # 澄清时附带该品类在售锚点：防止模型在"零商品知识"的澄清模式里
        # 谎称"不卖这个品类"（v2 GS-08"不卖床垫"幻觉）
        cat_hint = ""
        r = tools.search_products(categories[0], top_k=2, category=categories[0])
        if r.get("ok") and r["items"]:
            cat_hint = ("\n店内该品类在售（可作为方向锚点提及，绝不能说店里没有）："
                        + "、".join(i["name"] for i in r["items"]))
        parts.append({
            "intent": itype,
            "instruction": GUIDE_CLARIFY_PROMPT.replace(PERSONA, "").strip() +
                f"\n本次只问：{[factor_names.get(f, f) for f in missing][:2]}",
            "context": f"已收集的需求画像：{json.dumps(profile, ensure_ascii=False)}{cat_hint}",
        })
        return {"profile": profile, "clarify_rounds": clarify_rounds + 1}

    # 单类导购：识别出品类的按品类限定检索，识别不出的全库检索
    query = f"{text} {json.dumps(profile, ensure_ascii=False)}"
    if categories:
        found = tools.search_products(query, top_k=3, category=categories[0], size=psize)
        if not found.get("ok") or not found["items"]:
            found = tools.search_products(query, top_k=3, size=psize)
    else:
        found = tools.search_products(query, top_k=3, size=psize)
    return _recommend_parts(state, itype, found.get("items", []) if found.get("ok") else [],
                            parts, cards, profile)


def _recommend_parts(state: AgentState, itype: str, items: list[dict],
                     parts: list[dict], cards: list[dict], profile: dict,
                     extra_context: str = "") -> dict:
    """检索结果 → 推荐 part + 卡片 + 评价佐证（证据绑定：卡片与评价数据同源自检索结果）。"""
    if not items:
        parts.append({
            "intent": itype,
            "instruction": "检索无结果。如实告知暂时没有匹配的商品，并给出最接近的替代方向建议。",
            "context": "（无检索结果）",
        })
        return {"profile": profile, "clarify_rounds": 0}

    ids = [i["product_id"] for i in items]
    review_data = tools.analyze_reviews(ids)
    # 评价数据按商品名重排：模型按 P 编号映射商品名时容易张冠李戴（v3 GS-04 教训）
    name_by_id = {i["product_id"]: i["name"] for i in items}
    review_named = {name_by_id.get(pid, pid): v
                    for pid, v in review_data.get("summary", {}).items()}
    for item in items:
        cards.append({
            "product_id": item["product_id"], "name": item["name"],
            "price": item["price_range"][0], "reason": "、".join(item["tags"][:3]),
        })

    style = GUIDE_BUNDLE_PROMPT if itype == "guide_bundle" else GUIDE_RECOMMEND_PROMPT
    parts.append({
        "intent": itype,
        "instruction": style.replace(PERSONA, "").strip(),
        "context": (
            "候选商品：\n" + wrap_untrusted(json.dumps(items, ensure_ascii=False, indent=1), "products")
            + "\n\n评价分析（按商品名索引，数据可直接引用）：\n"
            + wrap_untrusted(json.dumps(review_named, ensure_ascii=False, indent=1), "reviews")
            + extra_context
        ),
    })
    # 证据随状态外发：评估 judge 据此核验回答中的数字（可验证的 grounding 检查）。
    # 必须覆盖回答可能引用的一切数字：评价统计 + 价格 + 颜色库存
    # （v3 教训：证据缺库存字段，judge 把真实的库存陈述误判为编造）
    return {"profile": profile, "clarify_rounds": 0,
            "evidence": {"reviews": review_named,
                         "prices": {i["name"]: i["price_range"] for i in items},
                         "color_prices": {i["name"]: {c["color"]: c["price"]
                                                      for c in i["colors"]}
                                          for i in items},
                         "stock": {i["name"]: {c["color"]: c["stock"] for c in i["colors"]}
                                   for i in items}}}


def _promo_context() -> str:
    """套装场景附带当前活动信息（GM-06 教训：夏凉套装漏答 2件9折 活动）。"""
    from app.services.retrieval import get_retriever
    hits = get_retriever().search("满减 折扣 优惠活动 专区", kind="knowledge", top_k=2)
    if not hits:
        return ""
    body = "\n".join(h["text"] for h in hits)
    return "\n\n当前店铺活动（推荐时主动提示相关优惠）：\n" + wrap_untrusted(body, "knowledge")


# ---------------------------------------------------------------- 客服 L1/L2 子逻辑
# 洗护类问题关键词：命中时强制附带知识库召回
# （v1 CARE-02/06 教训：商品 chunk 把洗护指南挤出 top_k，属"检索错"）
CARE_KEYWORDS = ("洗", "洗护", "清洗", "掉毛", "浮毛", "起球", "保养", "清洁", "晾晒", "烘干")


def _handle_consult(state: AgentState, intent: dict, parts: list[dict]) -> dict:
    """客服问答子逻辑。返回可能携带 evidence 的 updates（无证据时返回 {}）。

    证据外发（v5 终验教训）：商品咨询的回答会引用价格/库存数字，若不外发证据，
    评估 judge 无法核验，会把真实数字误判为编造（PROD-06/INV-01 稳定冤案）。
    """
    from app.services.retrieval import get_retriever

    # 检索查询以用户原话为主（保留"洗过""还能退"等关键语义），实体作补充
    # —— 仅用实体查询会丢失动作/条件语义，实测导致政策类问题召回错误条款
    text = state["messages"][-1].content
    # 多意图场景剥掉订单号，避免无意义 token 稀释政策检索（v1 MULTI-04 发票未召回）
    query = ORDER_ID_RE.sub("", text).strip()
    if not query:
        query = json.dumps(intent.get("entities", {}), ensure_ascii=False)

    # 多意图场景：全句检索会被无关分句稀释（v1 MULTI-04：订单分句把发票说明挤出
    # top_k）。按咨询关键词挑选最相关分句组成检索 query。
    if len(state.get("intents", [])) > 1:
        CONSULT_KW = ("退", "换", "退款", "发票", "优惠", "活动", "会员", "积分",
                      "运费", "发货", "快递", "差价", "政策", "洗", "材质", "尺寸",
                      "价格", "库存", "支", "含量", "颜色", "规格", "便宜", "贵")
        clauses = [c.strip() for c in re.split(r"[，。,？?！!；;]", query) if c.strip()]
        scored = sorted(
            ((sum(k in c for k in CONSULT_KW), c) for c in clauses),
            key=lambda x: -x[0],
        )
        picked = [c for s, c in scored if s > 0]
        if picked:
            query = " ".join(picked[:2])

    retriever = get_retriever()
    if intent["type"] == "consult_policy":
        results = retriever.search(query, kind="knowledge", top_k=4)
        ctx = "\n\n".join(wrap_untrusted(r["text"], "knowledge") for r in results)
        evidence: dict = {}
    else:
        # 商品咨询：同名多规格商品合并为一条上下文（规格/颜色库存全部列出）。
        # 两个 v1 教训一次性解决：
        # - INV-03"最便宜的被芯"：三条蚕丝被规格挤满 top_k，大豆纤维被未召回 → 按名分组
        # - PROD-06"有蚕丝被吗"：只答了部分规格 → 组内全规格合并展示
        raw = retriever.search(query, top_k=10)
        groups: dict[str, list[str]] = {}
        group_ranked: list[str] = []
        for r in raw:
            if r["kind"] == "product":
                base = r["payload"]["name"].split(" ")[0]
                if base not in groups:
                    groups[base] = []
                    group_ranked.append(base)
                groups[base].append(r["text"])
        # 按 rank 序交错输出：商品组（合并文本）与知识库 chunk，共取 5 条
        merged: list[str] = []
        shown_bases: set[str] = set()
        for r in raw:
            if len(merged) >= 5:
                break
            if r["kind"] == "product":
                base = r["payload"]["name"].split(" ")[0]
                if base in group_ranked:
                    merged.append(wrap_untrusted("\n".join(groups[base]), "knowledge"))
                    group_ranked.remove(base)
                    shown_bases.add(base)
            else:
                merged.append(wrap_untrusted(r["text"], "knowledge"))
        # 洗护类问题强制附带知识库召回（洗护指南可能被商品 chunk 挤出 top_k）
        if any(kw in text for kw in CARE_KEYWORDS):
            kb = retriever.search(query, kind="knowledge", top_k=2)
            for r in kb:
                block = wrap_untrusted(r["text"], "knowledge")
                if block not in merged:
                    merged.append(block)
        ctx = "\n\n".join(merged)
        # 证据：只覆盖进入上下文的商品（模型只能引用它看到的数字）
        prices: dict[str, list] = {}
        color_prices: dict[str, dict] = {}
        stock: dict[str, dict] = {}
        for r in raw:
            if r["kind"] != "product":
                continue
            p = r["payload"]
            if p["name"].split(" ")[0] not in shown_bases:
                continue
            prices[p["name"]] = [min(s["price"] for s in p["skus"]),
                                 max(s["price"] for s in p["skus"])]
            color_prices[p["name"]] = {s["color"]: s["price"] for s in p["skus"]}
            stock[p["name"]] = {s["color"]: s["stock"] for s in p["skus"]}
        evidence = ({"prices": prices, "color_prices": color_prices, "stock": stock}
                    if prices else {})
    parts.append({
        "intent": intent["type"],
        "instruction": QA_PROMPT.replace(PERSONA, "").strip(),
        "context": ctx or "（知识库未命中，如实告知需人工确认）",
    })
    return {"evidence": evidence} if evidence else {}


ORDER_ID_RE = re.compile(r"DD\d{11}")

# 无操作能力的请求：说明边界并引导人工，绝不暗示已代办（v1 ORD-07 教训）
NO_CAPABILITY_RULE = (
    "特别注意：你没有改地址、退款、修改订单等操作能力。"
    "用户提出这类请求时，明确说明小织无法直接办理、引导其联系人工客服处理，"
    "绝不能说'帮您试试更改''已为您修改'这类暗示可以代办的表述。"
)


def _handle_order(state: AgentState, parts: list[dict]) -> None:
    from app.services.retrieval import get_retriever

    text = state["messages"][-1].content
    m = ORDER_ID_RE.search(text)
    if not m:
        # 问物流异常处理规则但没给订单号：先答规则，再请用户提供订单号
        # （v1 POL-10 教训：只索取订单号漏答了用户真正问的处理规则）
        if any(kw in text for kw in ("没更新", "不更新", "异常", "太慢", "物流问题")):
            hits = get_retriever().search("物流异常 无更新 核查 处理方案", kind="knowledge", top_k=1)
            policy = hits[0]["text"] if hits else ""
            parts.append({
                "intent": "query_order",
                "instruction": "用户在问物流异常的处理规则。先按提供的政策内容回答处理规则，"
                               "再请用户提供订单号（DD 开头，可在'我的订单'里复制）以便具体跟进。",
                "context": wrap_untrusted(policy, "knowledge") if policy else "（缺订单号）",
            })
            return
        parts.append({
            "intent": "query_order",
            "instruction": "用户想查订单/物流但没给订单号。请用户提供订单号（DD 开头），"
                           "并告知可在'我的订单'里复制。" + NO_CAPABILITY_RULE,
            "context": "（缺订单号）",
        })
        return
    result = tools.query_logistics(m.group(0))
    instruction = ("基于工具返回的订单/物流信息如实回答，"
                   "已发货必含四要素：快递公司 + 单号 + 最新节点 + 节点时间"
                   "（如'中通 ZT7856…，8月5日 10:20 上海静安网点派送中'）；"
                   "只陈述工具返回的信息，不编造未返回的细节。"
                   + NO_CAPABILITY_RULE)
    ctx = wrap_untrusted(json.dumps(result, ensure_ascii=False), "order_data")
    if result.get("ok") and not result.get("logistics"):
        # 待发货订单必须附带发货时效政策，回答才有依据
        # （v1 ORD-02 教训：模型把大促 48 小时当成常规时效回答，属"生成错"）
        hits = get_retriever().search("发货时效 工作日 当天发 次日发", kind="knowledge", top_k=1)
        if hits:
            ctx += "\n\n发货时效政策：\n" + wrap_untrusted(hits[0]["text"], "knowledge")
        instruction += ("订单未发货：结合 created_at（下单时间）和发货时效政策，"
                        "明确告诉用户预计什么时候发出（如'昨晚9点付款，按规则次日发，"
                        "也就是今天会发出'），不要只背规则。")
    parts.append({
        "intent": "query_order",
        "instruction": instruction,
        "context": ctx,
    })
    # 订单数据同步进证据：judge 据此核验物流细节（v4 教训：judge 看不到订单数据，
    # 把真实的"王师傅派送中"误判为编造）
    return {"evidence": {"order": result}}


def _merge_evidence(a: dict, b: dict) -> dict:
    """证据深合并：reviews/prices/stock 等子字典按键合并，避免多意图场景互相覆盖。"""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


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
            guide_updates = _handle_guide(state, intent, parts, cards)
            # 证据合并：导购证据与订单证据可能同轮出现，不能互相覆盖
            if guide_updates.get("evidence") and updates.get("evidence"):
                guide_updates["evidence"] = _merge_evidence(updates["evidence"],
                                                            guide_updates["evidence"])
            updates.update(guide_updates)
        elif itype in {"consult_product", "consult_policy"}:
            # 客服问答证据（价格/库存）与导购/订单证据合并，不互相覆盖
            consult_updates = _handle_consult(state, intent, parts)
            if consult_updates.get("evidence"):
                updates["evidence"] = _merge_evidence(updates.get("evidence", {}),
                                                      consult_updates["evidence"])
        elif itype == "query_order":
            # 订单分支返回证据（物流/下单时间），与导购证据合并而非覆盖
            order_updates = _handle_order(state, parts) or {}
            if order_updates.get("evidence"):
                merged_evidence = _merge_evidence(updates.get("evidence", {}),
                                                  order_updates["evidence"])
                updates["evidence"] = merged_evidence
        elif itype == "chat":
            parts.append({
                "intent": "chat",
                "instruction": "友好简短回应闲聊（一句话），然后自然引导回家纺业务。"
                               "没有实时能力的问题（天气/新闻/股票），先说明小织不清楚、"
                               "绝不编造具体天气或温度；对方要笑话可以坦率说不太擅长，"
                               "也可以用一句家纺相关的小幽默接住，再引导回业务。",
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
