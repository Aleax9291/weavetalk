"""评估运行器：批量跑评估集 → 确定性检查 + LLM judge → 评审报告。

用法（服务需已启动）：
    uv run python -X utf8 scripts/run_eval.py [--tag v1]

双层判定：
1. 确定性检查（零成本）：意图匹配 / 转人工触发 / 必含要素 / 禁区词
2. LLM judge：对照参考要点判语义对错，失败时做四分归因（检索错/理解错/生成错/策略错）

产出 eval/report-<tag>.md：汇总指标 + 每题详情 + 人工评分留空栏。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import requests
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv()
BASE_URL = os.environ["LLM_BASE_URL"]
API_KEY = os.environ["LLM_API_KEY"]
MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

BASE = "http://127.0.0.1:8000"
EVAL_SET = Path(__file__).resolve().parent.parent / "eval" / "eval_set.json"

JUDGE_PROMPT = """你是严格的家纺客服智能体评估员。给定用户问题、参考答案要点、智能体实际回答，判定回答是否合格。

合格标准：
1. 事实正确：与参考答案要点一致，不编造政策/参数/库存/订单状态
2. 行为恰当：该澄清的澄清、该推荐的推荐、该拒绝的拒绝、该转人工的不自作主张
3. 不承诺系统没有的能力（如看图片、改地址、退款项）

输出 JSON：{"pass": true/false, "reason": "一句话理由", "attribution": "检索错|理解错|生成错|策略错|null"}
attribution 仅在 pass=false 时填写：
- 检索错：回答用的政策/商品信息与实际知识库不符（信息本身错）
- 理解错：误解用户问题意图或漏答部分问题
- 生成错：信息都在但表述错误/自相矛盾/编造细节
- 策略错：行为决策错（该澄清没澄清/该转人工没转/推荐了不该推荐的）"""


def run_case(client: requests.Session, case: dict) -> dict:
    """单题执行：独立会话，收集回复/意图/转人工/卡片。"""
    sid = client.post(f"{BASE}/api/session", timeout=10).json()["session_id"]
    resp = client.post(
        f"{BASE}/api/chat",
        json={"message": case["q"], "session_id": sid},
        stream=True, timeout=120,
    )
    text, intents, handoff, cards = "", [], False, []
    event = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].removeprefix(" ")
            if event == "token":
                text += data
            elif event == "intent":
                intents = [i["type"] for i in json.loads(data)]
            elif event == "handoff":
                handoff = True
            elif event == "card":
                cards.append(json.loads(data).get("name", ""))
    return {"reply": text, "intents": intents, "handoff": handoff, "cards": cards}


def deterministic_checks(case: dict, result: dict) -> list[str]:
    """返回失败原因列表（空=通过）。"""
    fails: list[str] = []
    got = set(result["intents"])
    expect = case.get("intents", [])
    if expect == ["_multi"]:
        if len(result["intents"]) < 2:
            fails.append(f"意图检查：期望多意图，实际 {result['intents']}")
    elif expect and not got & set(expect):
        fails.append(f"意图检查：期望 {expect} 之一，实际 {result['intents']}")

    if case.get("handoff") is True and not result["handoff"]:
        fails.append("转人工检查：期望转人工但未触发")
    if case.get("handoff") is False and result["handoff"]:
        fails.append("转人工检查：不应转人工但触发了")

    reply = result["reply"]
    for group in case.get("inc", []):
        if not any(alt in reply for alt in group):
            fails.append(f"必含要素缺失：{group}")
    for bad in case.get("exc", []):
        if bad in reply:
            fails.append(f"命中禁区：{bad!r}")
    return fails


JUDGE_CONCURRENCY = 8


async def judge_all(cases: list[dict], results: list[dict]) -> list[dict]:
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)

    async def judge_one(case: dict, result: dict) -> dict:
        user = (
            f"【用户问题】{case['q']}\n"
            f"【参考答案要点】{case['ref']}\n"
            f"【智能体实际回答】{result['reply']}\n"
            f"【附带商品卡片】{result['cards'] or '无'}"
        )
        async with sem:
            for _ in range(2):
                try:
                    r = await client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "system", "content": JUDGE_PROMPT},
                                  {"role": "user", "content": user}],
                        temperature=0,
                        response_format={"type": "json_object"},
                    )
                    return json.loads(r.choices[0].message.content)
                except Exception:
                    await asyncio.sleep(2)
            return {"pass": None, "reason": "judge调用失败", "attribution": None}

    return await asyncio.gather(*[judge_one(c, r) for c, r in zip(cases, results)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--only-fail-judge", action="store_true",
                        help="只对确定性检查失败的题跑 judge（省钱模式）")
    args = parser.parse_args()

    cases = json.loads(EVAL_SET.read_text("utf-8"))["cases"]
    print(f"评估集：{len(cases)} 题")

    client = requests.Session()
    results, det_fails = [], []
    t0 = time.time()
    for i, case in enumerate(cases):
        try:
            result = run_case(client, case)
        except Exception as e:
            result = {"reply": f"（请求失败：{e}）", "intents": [], "handoff": False, "cards": []}
        fails = deterministic_checks(case, result)
        results.append(result)
        det_fails.append(fails)
        mark = "✓" if not fails else "✗"
        print(f"[{i+1}/{len(cases)}] {mark} {case['id']} {case['q'][:20]}... "
              f"({time.time()-t0:.0f}s)")

    print("\nLLM judge 评审中...")
    if args.only_fail_judge:
        verdicts = [
            ({"pass": None, "reason": "确定性检查通过，未评审", "attribution": None}
             if not f else None)
            for f in det_fails
        ]
        need = [i for i, v in enumerate(verdicts) if v is None]
        judged = asyncio.run(judge_all([cases[i] for i in need], [results[i] for i in need]))
        for i, v in zip(need, judged):
            verdicts[i] = v
    else:
        verdicts = asyncio.run(judge_all(cases, results))

    # 汇总：最终判定 = 确定性检查 + judge 双通过
    rows = []
    n_pass = 0
    by_cat: dict[str, list[bool]] = {}
    for case, result, fails, verdict in zip(cases, results, det_fails, verdicts):
        judge_pass = verdict.get("pass")
        final = not fails and judge_pass is not False
        n_pass += final
        by_cat.setdefault(case["cat"], []).append(final)
        rows.append({
            "case": case, "result": result, "det_fails": fails,
            "verdict": verdict, "final": final,
        })

    report = render_report(rows, n_pass, len(cases), by_cat, args.tag,
                           elapsed=time.time() - t0)
    out = Path(__file__).resolve().parent.parent / "eval" / f"report-{args.tag}.md"
    out.write_text(report, "utf-8")
    print(f"\n通过率 {n_pass}/{len(cases)} ({n_pass/len(cases)*100:.1f}%)")
    print(f"报告：{out}")


def render_report(rows: list[dict], n_pass: int, total: int,
                  by_cat: dict[str, list[bool]], tag: str, elapsed: float) -> str:
    lines = [
        f"# 织语 WeaveTalk 评估报告（{tag}）",
        "",
        f"- 总通过率：**{n_pass}/{total}（{n_pass/total*100:.1f}%）**",
        f"- 耗时：{elapsed/60:.1f} 分钟",
        f"- 判定 = 确定性检查（意图/转人工/必含/禁区）+ LLM judge（语义对照参考要点）",
        "",
        "## 分类通过率",
        "",
        "| 类别 | 通过率 |",
        "|---|---|",
    ]
    for cat, flags in by_cat.items():
        p = sum(flags)
        lines.append(f"| {cat} | {p}/{len(flags)}（{p/len(flags)*100:.0f}%） |")
    lines += [
        "",
        "## 明细（人工评审请在最后一列打分：✓/✗/△ + 备注）",
        "",
    ]
    for row in rows:
        c, r, v = row["case"], row["result"], row["verdict"]
        status = "✅" if row["final"] else "❌"
        lines += [
            f"### {status} {c['id']}【{c['cat']}】{c['q']}",
            "",
            f"- **参考要点**：{c['ref']}",
            f"- **Agent 回答**：{r['reply']}",
            f"- **意图**：{r['intents']}　**转人工**：{r['handoff']}　**卡片**：{r['cards'] or '无'}",
        ]
        if row["det_fails"]:
            lines.append(f"- **确定性检查失败**：{'；'.join(row['det_fails'])}")
        lines.append(
            f"- **judge**：{'通过' if v.get('pass') else '不通过'}"
            f"（{v.get('reason', '')}）归因：{v.get('attribution') or '-'}"
        )
        lines += ["- **人工评分**：", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
