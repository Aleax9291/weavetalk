"""会话状态定义（LangGraph State）。

承载三类信息：
1. 对话历史（messages）
2. 路由结果（intents：多意图列表 + 当前处理指针）
3. 导购上下文（profile：澄清过程中收集的用户需求画像）
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class Intent(TypedDict, total=False):
    type: str            # 意图类型，见 docs/00-意图体系.md
    confidence: float
    entities: dict       # 抽取的实体（商品/订单号/尺寸等）


class ShoppingProfile(TypedDict, total=False):
    """导购澄清中逐步填充的用户需求画像（家纺品类决策树）。"""
    bed_size: str        # 床型尺寸：1.5m / 1.8m / 2.0m
    material: str        # 材质偏好：全棉 / 天丝 / 真丝 / 磨毛
    season: str          # 季节/温感：夏凉 / 冬暖 / 四季
    sleep_habit: str     # 睡眠习惯：怕冷 / 怕热 / 敏感肌 / 颈椎需求
    budget: str          # 预算区间
    purpose: str         # 用途：自用 / 送礼 / 婚庆 / 宿舍


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    intents: list[Intent]          # 本轮识别出的多意图列表（按优先级排序）
    pending: list[Intent]          # 挂起的未处理意图（"还有呢"式追问）
    need_clarify: bool             # 主意图置信度不足，需澄清
    emotion: str                   # calm / confused / anxious / angry
    profile: ShoppingProfile       # 导购需求画像
    clarify_rounds: int            # 本轮导购已澄清轮数（上限 3）
    handoff: bool                  # 是否触发转人工
    handoff_reason: str
    answer_parts: list[dict]       # 各意图的处理产出（instruction + context）
    cards: list[dict]              # 待展示的商品卡片
    gen_messages: list[dict]       # 汇总节点产出的最终生成消息（端点流式输出）
    answer_hint: str               # 汇总节点给最终生成的整合指令
    final_text: str                # 确定性回复（转人工/攻击兜底，跳过长文本生成）
    evidence: dict                 # 推荐证据（评价聚合+候选价格），随 SSE 外发给评估 judge 核验数字
    trace: list[dict]              # 全链路轨迹（评估归因用）
