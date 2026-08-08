"""业务工具：商品检索 / 评价分析 / 订单 / 物流 / 库存 / 转人工。

设计要点：
1. 评价分析是**确定性聚合**（不是 LLM 生成）——推荐理由引用的每个数字
   都直接来自 reviews.json 统计，GroundingGuard 可逐条验证，零幻觉
2. 身份参数（shop_id/session_id）由服务端注入，模型传值忽略（见 security.py）
3. 工具异常收敛为友好错误字符串，不抛栈给模型（吸收 Customer-Agent 降级设计）
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from app.services.retrieval import get_retriever

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load(name: str) -> list[dict]:
    return json.loads((DATA_DIR / name).read_text("utf-8"))


# ---------------------------------------------------------------- 商品检索
def search_products(query: str, *, top_k: int = 5, in_stock_only: bool = True) -> dict:
    """混合检索商品，默认过滤全缺货商品。"""
    try:
        results = get_retriever().search(query, kind="product", top_k=top_k * 2)
        items = []
        for r in results:
            p = r["payload"]
            skus = [s for s in p["skus"] if s["stock"] > 0] if in_stock_only else p["skus"]
            if not skus:
                continue
            items.append({
                "product_id": p["product_id"], "name": p["name"],
                "category": p["category"], "material": p["material"],
                "spec": p["spec"], "tags": p["tags"], "desc": p["desc"],
                "price_range": [min(s["price"] for s in skus), max(s["price"] for s in skus)],
                "low_stock": any(0 < s["stock"] <= 8 for s in skus),
                "sales": p["sales"], "score": r["score"],
            })
            if len(items) >= top_k:
                break
        return {"ok": True, "items": items}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"search_products failed: {e}")
        return {"ok": False, "error": "商品检索暂时不可用，请稍后重试"}


# ---------------------------------------------------------------- 评价分析（确定性聚合）
def analyze_reviews(product_ids: list[str]) -> dict:
    """聚合商品评价：好评点/差评点占比 + 代表性原声。全部数字可溯源。"""
    try:
        reviews = [r for r in _load("reviews.json") if r["product_id"] in product_ids]
        if not reviews:
            return {"ok": True, "summary": {}, "msg": "暂无评价数据"}
        out: dict = {}
        for pid in product_ids:
            rs = [r for r in reviews if r["product_id"] == pid]
            if not rs:
                continue
            pos: dict[str, list[str]] = {}
            neg: dict[str, list[str]] = {}
            for r in rs:
                if r["sentiment"] == "positive" and r["aspect"]:
                    pos.setdefault(r["aspect"], []).append(r["content"])
                elif r["sentiment"] == "negative" and r["aspect"]:
                    neg.setdefault(r["aspect"], []).append(r["content"])
            total = len(rs)
            out[pid] = {
                "total": total,
                "positive_rate": round(sum(1 for r in rs if r["rating"] >= 4) / total, 2),
                "pros": [
                    {"aspect": a, "ratio": round(len(q) / total, 2), "quote": q[0]}
                    for a, q in sorted(pos.items(), key=lambda x: -len(x[1]))[:3]
                ],
                "cons": [
                    {"aspect": a, "ratio": round(len(q) / total, 2), "quote": q[0]}
                    for a, q in sorted(neg.items(), key=lambda x: -len(x[1]))[:3]
                ],
            }
        return {"ok": True, "summary": out}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"analyze_reviews failed: {e}")
        return {"ok": False, "error": "评价分析暂时不可用"}


# ---------------------------------------------------------------- L2 订单类
def query_order(order_id: str) -> dict:
    for o in _load("mock_orders.json"):
        if o["order_id"] == order_id:
            return {"ok": True, "order": o}
    return {"ok": False, "error": f"没有找到订单 {order_id}，请核对订单号"}


def query_logistics(order_id: str) -> dict:
    for o in _load("mock_orders.json"):
        if o["order_id"] == order_id:
            logi = o.get("logistics", {})
            if not logi.get("tracking_no"):
                return {"ok": True, "msg": "订单暂未发货，暂无物流信息", "status": o["status"]}
            return {"ok": True, "status": o["status"], "logistics": logi}
    return {"ok": False, "error": f"没有找到订单 {order_id}，请核对订单号"}


def check_inventory(product_id: str) -> dict:
    for p in _load("products.json"):
        if p["product_id"] == product_id:
            return {
                "ok": True, "name": p["name"],
                "skus": [
                    {"sku_id": s["sku_id"], "color": s["color"],
                     "stock": s["stock"], "price": s["price"]}
                    for s in p["skus"]
                ],
            }
    return {"ok": False, "error": "商品不存在"}
