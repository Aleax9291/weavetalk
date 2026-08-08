"""虚拟家纺店铺数据集生成器。

生成四类数据：
1. products.json    商品库（6 条产品线 × 多 SKU：尺寸×颜色，含缺货/多规格）
2. reviews.json     评价数据（每商品 30-80 条，按产品线特征分布褒贬，含差评关键词）
3. mock_orders.json mock 订单（含物流轨迹，供 L2 工具调用演示）
4. app/data/knowledge/*.md  知识库文档（政策/洗护/会员，供 L1 RAG）

数据集按真实店铺结构构造：有差评、有缺货、有价格波动——避免"玩具感"。
运行：uv run python scripts/build_dataset.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)

DATA = Path(__file__).resolve().parent.parent / "app" / "data"
(DATA / "knowledge").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 商品线定义
# 每条产品线带"关键决策因子"（导购链路用）和"评价特征"（评价分析工具用）
PRODUCT_LINES = [
    {
        "line_id": "sijiantao-qm", "category": "四件套", "name": "全棉长绒棉四件套",
        "material": "全棉", "price_range": (259, 399),
        "key_factors": ["bed_size"],
        "tags": ["60支", "A类母婴级", "新疆长绒棉"],
        "pros": [("柔软", 0.87), ("亲肤", 0.72), ("不掉色", 0.65), ("透气", 0.58)],
        "cons": [("初次洗轻微浮色", 0.12), ("偏薄", 0.08)],
        "desc": "60支新疆长绒棉，A类母婴级安全标准，裸睡级亲肤",
    },
    {
        "line_id": "sijiantao-ts", "category": "四件套", "name": "兰精天丝四件套",
        "material": "天丝", "price_range": (459, 659),
        "key_factors": ["bed_size"],
        "tags": ["100支", "莱赛尔", "凉感"],
        "pros": [("丝滑凉感", 0.91), ("垂感好", 0.76), ("颜值高", 0.69)],
        "cons": [("容易勾丝", 0.15), ("需手洗麻烦", 0.11)],
        "desc": "100支奥地利兰精天丝，夏季凉感首选，真丝般垂坠",
    },
    {
        "line_id": "sijiantao-mm", "category": "四件套", "name": "全棉磨毛四件套",
        "material": "磨毛", "price_range": (299, 429),
        "key_factors": ["bed_size"],
        "tags": ["秋冬", "加厚", "速暖"],
        "pros": [("暖和", 0.93), ("绒毛舒服", 0.81), ("不掉毛", 0.62)],
        "cons": [("轻微掉毛", 0.18), ("比预期厚", 0.07)],
        "desc": "全棉磨毛工艺，一触即暖，秋冬怕冷星人福音",
    },
    {
        "line_id": "beixin-sc", "category": "被芯", "name": "桑蚕丝被",
        "material": "蚕丝", "price_range": (899, 1599),
        "key_factors": ["bed_size", "season", "sleep_habit"],
        "tags": ["100%桑蚕丝", "子母被", "四季通用"],
        "pros": [("轻", 0.89), ("贴身不压身", 0.84), ("恒温", 0.71)],
        "cons": [("不能水洗", 0.13), ("价格高", 0.09)],
        "desc": "100%双宫桑蚕丝长丝，子母被设计四季可用",
    },
    {
        "line_id": "beixin-dd", "category": "被芯", "name": "大豆纤维被",
        "material": "大豆纤维", "price_range": (199, 359),
        "key_factors": ["bed_size", "season", "sleep_habit"],
        "tags": ["可机洗", "性价比", "抗菌"],
        "pros": [("可机洗方便", 0.92), ("性价比", 0.85), ("蓬松", 0.67)],
        "cons": [("久用略塌", 0.16), ("新被有味道需晾", 0.10)],
        "desc": "大豆蛋白纤维，整被可机洗，学生党/租房党首选",
    },
    {
        "line_id": "beixin-yr", "category": "被芯", "name": "95白鹅绒被",
        "material": "羽绒", "price_range": (1299, 2199),
        "key_factors": ["bed_size", "season", "sleep_habit"],
        "tags": ["95%绒子含量", "800+蓬松度", "静音面料"],
        "pros": [("轻暖", 0.94), ("蓬松", 0.88), ("不跑绒", 0.73)],
        "cons": [("偶有轻微响声", 0.08), ("贵", 0.11)],
        "desc": "95%东北白鹅绒，800+蓬松度，-15℃深冬无忧",
    },
    {
        "line_id": "zhenxin-rj", "category": "枕芯", "name": "泰国乳胶枕",
        "material": "乳胶", "price_range": (159, 269),
        "key_factors": ["sleep_habit"],
        "tags": ["93%乳胶含量", "波浪曲线", "护颈"],
        "pros": [("支撑好", 0.86), ("颈椎舒服", 0.79), ("回弹快", 0.71)],
        "cons": [("乳胶味需散几天", 0.19), ("偏高不习惯", 0.12)],
        "desc": "93%泰国天然乳胶，波浪曲线设计，侧睡仰睡双区支撑",
    },
    {
        "line_id": "zhenxin-yr", "category": "枕芯", "name": "羽绒枕",
        "material": "羽绒", "price_range": (329, 499),
        "key_factors": ["sleep_habit"],
        "tags": ["三层立体", "可水洗", "酒店同款"],
        "pros": [("软", 0.88), ("包裹感", 0.82), ("酒店感", 0.75)],
        "cons": [("支撑偏弱", 0.14), ("需常拍打", 0.09)],
        "desc": "三层立体羽绒枕，五星酒店同款睡感，可整枕水洗",
    },
    {
        "line_id": "tan-cj", "category": "毯类", "name": "新疆棉纱布盖毯",
        "material": "全棉", "price_range": (99, 169),
        "key_factors": ["season"],
        "tags": ["六层纱布", "空调毯", "A类"],
        "pros": [("透气", 0.90), ("越洗越软", 0.78), ("轻便", 0.72)],
        "cons": [("颜色选择少", 0.08)],
        "desc": "六层新疆棉纱布，空调房/沙发/办公室多场景",
    },
    {
        "line_id": "chuangdian-rj", "category": "床垫", "name": "乳胶椰棕床垫",
        "material": "乳胶椰棕", "price_range": (799, 1399),
        "key_factors": ["bed_size", "sleep_habit"],
        "tags": ["软硬两用", "护脊", "可拆洗"],
        "pros": [("护腰", 0.85), ("软硬两面实用", 0.80), ("无异味", 0.74)],
        "cons": [("偏重搬运费劲", 0.13)],
        "desc": "乳胶+椰棕双面设计，一面偏软一面护脊，外套可拆洗",
    },
]

SIZES = {"1.5m床": "200×230cm", "1.8m床": "220×240cm", "2.0m床": "240×260cm"}
COLORS = ["奶白", "浅灰", "雾蓝", "豆绿", "藕粉", "驼色"]
BED_CATS = {"四件套", "床垫"}

REVIEW_TPL_PRO = [
    "用了半个月来评价，{point}确实不错，会回购",
    "{point}！这个价格能买到真的惊喜",
    "给家里买的，爸妈说{point}，很满意",
    "第二次买了，{point}一如既往",
    "对比了好几家才下单，{point}没让我失望",
    "实物比图片还好，{point}是真的",
    "睡了一周，最大的感受就是{point}",
]
REVIEW_TPL_CON = [
    "整体还行，就是{point}，介意慎拍",
    "{point}这个问题确实存在，希望改进",
    "东西不错但{point}，扣一星",
    "用了一个月，{point}，有点小失望",
    "客服说正常，但{point}还是影响体验",
]
REVIEW_TPL_MID = [
    "中规中矩，对得起价格",
    "还可以，没有特别惊艳但也不差",
    "物流很快，东西符合描述",
]

# ---------------------------------------------------------------- 生成商品库
products = []
pid = 1000
for line in PRODUCT_LINES:
    sizes = list(SIZES.items()) if line["category"] in BED_CATS or line["category"] == "被芯" else [("均码", "标准")]
    for size_name, size_spec in sizes:
        pid += 1
        base_price = random.randint(*line["price_range"])
        if size_name == "1.8m床":
            base_price = int(base_price * 1.15)
        elif size_name == "2.0m床":
            base_price = int(base_price * 1.3)
        skus = []
        for i, color in enumerate(random.sample(COLORS, 4)):
            stock = random.choices(
                [0, random.randint(2, 8), random.randint(9, 50), random.randint(51, 200)],
                weights=[0.1, 0.15, 0.35, 0.4],  # 缺货仅 ~10%，另有低库存 SKU 供"库存紧张"促单演示
            )[0]
            skus.append({
                "sku_id": f"SKU{pid}{i:02d}", "color": color, "stock": stock,
                "price": base_price + random.choice([0, 0, 20, -10]),
            })
        products.append({
            "product_id": f"P{pid}", "line_id": line["line_id"],
            "category": line["category"], "name": f"{line['name']} {size_name}",
            "material": line["material"], "size": size_name, "spec": size_spec,
            "tags": line["tags"], "desc": line["desc"],
            "key_factors": line["key_factors"], "skus": skus,
            "sales": random.randint(50, 5000),
        })

(DATA / "products.json").write_text(
    json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------------------------------------------- 生成评价
reviews = []
rid = 10000
line_map = {l["line_id"]: l for l in PRODUCT_LINES}
for p in products:
    line = line_map[p["line_id"]]
    n = random.randint(30, 80)
    for _ in range(n):
        rid += 1
        roll = random.random()
        if roll < 0.72:
            point, _ = random.choices(line["pros"], weights=[w for _, w in line["pros"]])[0]
            content = random.choice(REVIEW_TPL_PRO).format(point=point)
            rating = random.choice([5, 5, 5, 4])
            aspect, sentiment = point, "positive"
        elif roll < 0.88:
            point, _ = random.choices(line["cons"], weights=[w for _, w in line["cons"]])[0]
            content = random.choice(REVIEW_TPL_CON).format(point=point)
            rating = random.choice([3, 2, 3])
            aspect, sentiment = point, "negative"
        else:
            content = random.choice(REVIEW_TPL_MID)
            rating = random.choice([4, 3])
            aspect, sentiment = None, "neutral"
        reviews.append({
            "review_id": f"R{rid}", "product_id": p["product_id"],
            "rating": rating, "content": content,
            "aspect": aspect, "sentiment": sentiment,
        })

(DATA / "reviews.json").write_text(
    json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------------------------------------------- 生成 mock 订单
orders = [
    {
        "order_id": "DD20260801001", "status": "已发货",
        "items": [{"name": "兰精天丝四件套 1.8m床 雾蓝", "qty": 1, "price": 559}],
        "total": 559, "created_at": "2026-08-01 14:23",
        "logistics": {
            "company": "中通快递", "tracking_no": "ZT7856234109",
            "traces": [
                {"time": "2026-08-01 18:30", "desc": "【杭州】包裹已揽收"},
                {"time": "2026-08-02 09:12", "desc": "【杭州转运中心】已发出"},
                {"time": "2026-08-04 08:45", "desc": "【上海转运中心】到达"},
                {"time": "2026-08-05 10:20", "desc": "【上海静安网点】派送中，快递员：王师傅 138****6789"},
            ],
        },
    },
    {
        "order_id": "DD20260805002", "status": "待发货",
        "items": [{"name": "桑蚕丝被 1.8m床", "qty": 1, "price": 1299}],
        "total": 1299, "created_at": "2026-08-05 21:07",
        "logistics": {"company": None, "tracking_no": None, "traces": []},
    },
    {
        "order_id": "DD20260728003", "status": "已签收",
        "items": [
            {"name": "泰国乳胶枕", "qty": 2, "price": 189},
            {"name": "新疆棉纱布盖毯", "qty": 1, "price": 129},
        ],
        "total": 507, "created_at": "2026-07-28 10:15",
        "logistics": {
            "company": "顺丰速运", "tracking_no": "SF1345678902",
            "traces": [
                {"time": "2026-07-28 15:00", "desc": "【杭州】包裹已揽收"},
                {"time": "2026-07-30 11:32", "desc": "已签收，签收人：本人"},
            ],
        },
    },
]
(DATA / "mock_orders.json").write_text(
    json.dumps(orders, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------------------------------------------- 知识库文档
KNOWLEDGE = {
    "退换货政策.md": """# 退换货政策

## 七天无理由退货
- 自签收次日起 7 天内，商品未使用、未洗涤、吊牌包装完好可申请无理由退货
- 床品属贴身用品，**已洗涤、已使用的不支持无理由退货**（质量问题除外）
- 无理由退货运费由买家承担；质量问题退货运费由店铺承担

## 质量问题退换
- 签收 15 天内发现质量问题（掉色超标、起球、破损、填充物异常），支持免费退换
- 需提供问题部位照片，审核通过后安排上门取件
- 退款时效：退货签收后 48 小时内原路退回

## 换货
- 尺寸/颜色换货：在商品未使用前提下支持一次免费换货
- 换货差价多退少补
""",
    "物流与发货.md": """# 物流与发货

- 默认快递：中通/圆通，单笔满 299 元升级顺丰
- 发货时效：工作日 16:00 前付款当天发，其余次日发（大促期间 48 小时内）
- 新疆、西藏、内蒙古部分地区需补 10 元运费
- 支持指定快递：下单后联系客服备注，顺丰需补差价 8 元
- 物流异常（72 小时无更新）：客服将主动联系快递核查，24 小时内回复处理方案
""",
    "优惠与活动.md": """# 优惠与活动

## 当前活动（2026 年 8 月）
- 全店满 399 减 30，满 699 减 60（可叠加店铺券）
- 新客立减 15 元（首单专享）
- 夏凉专区：天丝/纱布品类 2 件 9 折

## 会员权益
- 银卡（累计消费满 500）：98 折 + 生日券 20 元
- 金卡（累计消费满 2000）：95 折 + 生日券 50 元 + 专属客服
- 积分：消费 1 元 = 1 积分，100 积分抵 1 元，单笔最高抵 20%
- 积分有效期：获得后 12 个月

## 价格保护
- 签收后 7 天内降价，可申请退差价（秒杀/清仓品除外）
""",
    "洗护指南.md": """# 床品洗护指南

## 通用
- 新床品建议先洗后用；深色浅色分开洗
- 水温不超过 30℃，反面洗涤，阴凉通风处晾干，避免暴晒

## 分材质
- **全棉/磨毛**：可机洗，轻柔模式；磨毛款前两次洗有轻微浮毛属正常
- **天丝**：建议手洗或真丝模式，中性洗涤剂，不可拧绞、不可烘干
- **真丝**：必须手洗或干洗，专用丝毛洗涤剂
- **蚕丝被**：不可水洗不可干洗，定期通风晾晒（避免暴晒），局部污渍湿布擦拭
- **羽绒被/枕**：可机洗（大件模式），低温烘干加烘干球恢复蓬松
- **乳胶枕**：不可水洗不可暴晒，枕套勤洗，枕芯通风除味即可
""",
    "发票说明.md": """# 发票说明

- 支持电子普通发票，下单时或签收后 30 天内可申请
- 抬头：个人或企业（企业需提供税号）
- 开票内容：*纺织品*床上用品
- 补开：超过 30 天请联系客服人工处理
- 发票将在申请后 1-3 个工作日发送至预留邮箱
""",
}
for name, content in KNOWLEDGE.items():
    (DATA / "knowledge" / name).write_text(content, encoding="utf-8")

# ---------------------------------------------------------------- 汇总
n_sku = sum(len(p["skus"]) for p in products)
print(f"商品: {len(products)} 个（{n_sku} SKU），覆盖 {len(PRODUCT_LINES)} 条产品线")
print(f"评价: {len(reviews)} 条")
print(f"订单: {len(orders)} 个")
print(f"知识库: {len(KNOWLEDGE)} 篇")
