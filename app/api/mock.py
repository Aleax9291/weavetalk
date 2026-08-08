"""mock API：订单 / 物流 / 库存查询。

供 Agent 通过 Function Calling 调用（L2 工具可答链路），
数据来自 app/data/mock_orders.json，按真实电商接口返回结构构造。
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter()
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load(name: str) -> list[dict]:
    path = DATA_DIR / name
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/orders/{order_id}")
async def get_order(order_id: str) -> dict:
    for order in _load("mock_orders.json"):
        if order["order_id"] == order_id:
            return {"code": 0, "data": order}
    return {"code": 404, "msg": "订单不存在"}


@router.get("/logistics/{order_id}")
async def get_logistics(order_id: str) -> dict:
    for order in _load("mock_orders.json"):
        if order["order_id"] == order_id:
            return {"code": 0, "data": order.get("logistics", {})}
    return {"code": 404, "msg": "订单不存在"}


@router.get("/inventory/{sku_id}")
async def get_inventory(sku_id: str) -> dict:
    for product in _load("products.json"):
        for sku in product.get("skus", []):
            if sku["sku_id"] == sku_id:
                return {
                    "code": 0,
                    "data": {"sku_id": sku_id, "stock": sku["stock"]},
                }
    return {"code": 404, "msg": "SKU 不存在"}
