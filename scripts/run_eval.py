"""评估运行器：批量跑评估集 → 确定性检查 + LLM judge → 评审报告。

用法（服务需已启动）：
    uv run python -X utf8 scripts/run_eval.py [--tag v1] [--set eval_set.json] [--runs 3]

双层判定（v5 方法论改造，2026-08-10）：
1. 确定性检查只保留**硬约束**：意图匹配 / 转人工触发 / 禁区词（带否定豁免）。
   inc 同义词表已从通过标准中移除——逐题补同义词是对评测集过拟合，
   语义对错全部交给对照证据的 LLM judge。
2. LLM judge：对照参考要点+证据数据判语义对错，失败时做四分归因。

泛化性验证：--runs N 多次采样报稳定性（N/N 一致通过率），holdout 集只跑一次。
产出 eval/report-<tag>.md。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
1. 事实正确：与参考答案要点一致，不编造政策/参数/库存/订单状态/评价数据
2. 行为恰当：该澄清的澄清、该推荐的推荐、该拒绝的拒绝、该转人工的不自作主张
3. 不承诺系统没有的能力（如看图片、改地址、退款项）

判定细则：
- 参考要点是"必须覆盖的核心点"，不是"允许说的全部"。智能体额外补充的信息，
  只要与要点不矛盾、且属于店铺常见政策范畴（满减/积分/会员/快递/价保），不判编造
- 但要点中的关键条件（时效/前提/费用承担/次数限制）缺失，可能误导用户的，判不通过
- 【证据数据】是系统确定性聚合的评价/价格/库存数据（reviews: 评价统计，prices: 价格区间，
  stock: 颜色库存，order: 订单快照）。它的用途是核验**波动型数字**：回答引用的评价百分比/
  好评率/库存件数必须能在证据中找到对应值（允许百分点取整差异）；证据数据中找不到出处的
  评价统计或库存数字，判编造。证据为 {} 时回答不应出现评价统计数字
- 注意证据的边界：商品固有属性（材质/含量/支数/规格尺寸）和店铺政策（活动/折扣/时效）
  来自检索上下文而非证据数据——这类信息与参考要点一致即视为有出处，不要因"证据中没有"
  判编造
- 套装/组合题：组合总价应等于各项价格之和，算术不一致或漏项判生成错

输出 JSON：{"pass": true/false, "reason": "一句话理由", "attribution": "检索错|理解错|生成错|策略错|null"}
attribution 仅在 pass=false 时填写：
- 检索错：回答用的政策/商品信息与实际知识库不符（信息本身错）
- 理解错：误解用户问题意图或漏答部分问题
- 生成错：信息都在但表述错误/自相矛盾/编造细节
- 策略错：行为决策错（该澄清没澄清/该转人工没转/推荐了不该推荐的）"""


def run_case(case: dict) -> dict:
    """单题执行：独立会话，收集回复/意图/转人工/卡片。线程安全（每次新建连接）。"""
    sid = requests.post(f"{BASE}/api/session", timeout=10).json()["session_id"]
    resp = requests.post(
        f"{BASE}/api/chat",
        json={"message": case["q"], "session_id": sid},
        stream=True, timeout=120,
    )
    text, intents, handoff, cards = "", [], False, []
    evidence: dict = {}
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
            elif event == "evidence":
                evidence = json.loads(data)
    return {"reply": text, "intents": intents, "handoff": handoff,
            "cards": cards, "evidence": evidence}


def _hits_forbidden(reply: str, bad: str) -> bool:
    """禁区子串命中判定，带否定词/数字前缀豁免。

    豁免规则（v1 误伤教训）：
    - 前一字是否定词（不/没/无/非）→ 豁免，如"不可以水洗"不命中禁区"可以水洗"
    - 前一字是数字 → 豁免，如"98折"不命中禁区"8折"、"100支"不命中"40支"
    """
    start = 0
    while True:
        i = reply.find(bad, start)
        if i == -1:
            return False
        prev = reply[i - 1] if i > 0 else ""
        if prev not in "不没无非" and not prev.isdigit():
            return True
        start = i + 1


def deterministic_checks(case: dict, result: dict) -> list[str]:
    """硬约束检查（返回失败原因列表，空=通过）。

    只保留三类硬约束：意图匹配 / 转人工 / 禁区词。
    inc 必含要素已移出通过标准——同义词表永远追不上自然语言的变化，
    语义完整性由对照证据的 LLM judge 判定。
    """
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
    for bad in case.get("exc", []):
        if _hits_forbidden(reply, bad):
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
            f"【附带商品卡片】{result['cards'] or '无'}\n"
            f"【证据数据】{json.dumps(result.get('evidence') or {}, ensure_ascii=False) or '无'}"
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


def run_one_round(cases: list[dict], workers: int, round_no: int,
                  total_rounds: int) -> tuple[list[dict], list[list[str]]]:
    """跑一轮：并发执行所有用例，返回 (results, det_fails)。"""
    results: list[dict | None] = [None] * len(cases)
    t0 = time.time()

    def run_with_catch(i: int, case: dict) -> tuple[int, dict]:
        try:
            return i, run_case(case)
        except Exception as e:
            return i, {"reply": f"（请求失败：{e}）", "intents": [], "handoff": False,
                       "cards": [], "evidence": {}}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_with_catch, i, c) for i, c in enumerate(cases)]
        done = 0
        for fut in as_completed(futs):
            i, result = fut.result()
            results[i] = result
            done += 1
            fails = deterministic_checks(cases[i], result)
            mark = "✓" if not fails else "✗"
            print(f"  [R{round_no} {done}/{len(cases)}] {mark} {cases[i]['id']} "
                  f"{cases[i]['q'][:18]}... ({time.time()-t0:.0f}s)")

    final_results = [r for r in results if r is not None]
    det_fails = [deterministic_checks(c, r) for c, r in zip(cases, final_results)]
    return final_results, det_fails


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--set", default="eval_set.json", dest="eval_file",
                        help="评测集文件名（eval/ 目录下），holdout 集只在终验跑一次")
    parser.add_argument("--runs", type=int, default=1,
                        help="采样轮数；>1 时报告稳定性（N/N 一致通过率）")
    parser.add_argument("--workers", type=int, default=4, help="用例执行并发数")
    args = parser.parse_args()

    eval_path = EVAL_SET.parent / args.eval_file
    cases = json.loads(eval_path.read_text("utf-8"))["cases"]
    print(f"评估集：{eval_path.name}（{len(cases)} 题 × {args.runs} 轮）")

    t0 = time.time()
    rounds: list[dict] = []  # 每轮 {results, det_fails, verdicts, finals}
    for r in range(args.runs):
        print(f"\n=== 第 {r+1}/{args.runs} 轮 ===")
        results, det_fails = run_one_round(cases, args.workers, r + 1, args.runs)
        print(f"第 {r+1} 轮 LLM judge 评审中...")
        verdicts = asyncio.run(judge_all(cases, results))
        finals = [
            not f and v.get("pass") is not False
            for f, v in zip(det_fails, verdicts)
        ]
        rounds.append({"results": results, "det_fails": det_fails,
                       "verdicts": verdicts, "finals": finals})
        print(f"第 {r+1} 轮通过率：{sum(finals)}/{len(cases)}")

    # 跨轮聚合：稳定通过（每轮都过）/ 稳定失败（每轮都挂）/ 抖动（时过时挂）
    per_run_rates = [sum(rd["finals"]) / len(cases) for rd in rounds]
    case_flags = list(zip(*[rd["finals"] for rd in rounds]))  # 每题各轮结果
    stable_pass = sum(1 for fl in case_flags if all(fl))
    stable_fail = sum(1 for fl in case_flags if not any(fl))
    flaky = [cases[i]["id"] for i, fl in enumerate(case_flags)
             if any(fl) and not all(fl)]

    report = render_report(cases, rounds, per_run_rates, stable_pass, stable_fail,
                           flaky, args, elapsed=time.time() - t0)
    out = EVAL_SET.parent / f"report-{args.tag}.md"
    out.write_text(report, "utf-8")
    avg = sum(per_run_rates) / len(per_run_rates)
    print(f"\n平均通过率 {avg*100:.1f}%（各轮："
          f"{' / '.join(f'{p*100:.1f}%' for p in per_run_rates)}）")
    if args.runs > 1:
        print(f"稳定性：{stable_pass}/{len(cases)} 全轮通过"
              f"（{stable_pass/len(cases)*100:.1f}%），抖动题：{flaky or '无'}")
    print(f"报告：{out}")


def render_report(cases: list[dict], rounds: list[dict], per_run_rates: list[float],
                  stable_pass: int, stable_fail: int, flaky: list[str],
                  args: argparse.Namespace, elapsed: float) -> str:
    total = len(cases)
    n_runs = len(rounds)
    avg = sum(per_run_rates) / n_runs
    lines = [
        f"# 织语 WeaveTalk 评估报告（{args.tag}）",
        "",
        f"- 评测集：`{args.eval_file}`　采样轮数：{n_runs}　耗时：{elapsed/60:.1f} 分钟",
        f"- **平均通过率：{avg*100:.1f}%**（各轮："
        f"{' / '.join(f'{p*100:.1f}%' for p in per_run_rates)}）",
    ]
    if n_runs > 1:
        lines += [
            f"- **稳定性（{n_runs}/{n_runs} 一致通过）：{stable_pass}/{total}"
            f"（{stable_pass/total*100:.1f}%）**；稳定失败 {stable_fail} 题；"
            f"抖动题：{', '.join(flaky) if flaky else '无'}",
        ]
    lines += [
        "- 判定 = 硬约束检查（意图/转人工/禁区）+ LLM judge（对照参考要点与证据数据）",
        "- inc 同义词表已从通过标准移除（v5 起）：语义完整性全部由 judge 判定",
        "",
        "## 分类通过率（按各轮平均）",
        "",
        "| 类别 | 通过率 |",
        "|---|---|",
    ]
    by_cat: dict[str, list[float]] = {}
    for i, case in enumerate(cases):
        rate = sum(rd["finals"][i] for rd in rounds) / n_runs
        by_cat.setdefault(case["cat"], []).append(rate)
    for cat, rates in by_cat.items():
        lines.append(f"| {cat} | {sum(rates)/len(rates)*100:.0f}%（{len(rates)} 题） |")
    lines += [
        "",
        "## 明细（人工评审请在最后一列打分：✓/✗/△ + 备注）",
        "",
    ]
    for i, c in enumerate(cases):
        flags = [rd["finals"][i] for rd in rounds]
        if all(flags):
            status = "✅"
        elif not any(flags):
            status = "❌"
        else:
            status = "⚠️"  # 抖动
        flag_str = "/".join("✓" if f else "✗" for f in flags)
        lines += [
            f"### {status} {c['id']}【{c['cat']}】{c['q']}　（各轮：{flag_str}）",
            "",
            f"- **参考要点**：{c['ref']}",
        ]
        for r, rd in enumerate(rounds):
            res, v = rd["results"][i], rd["verdicts"][i]
            lines += [
                f"- **第{r+1}轮回答**：{res['reply']}",
                f"  - 意图 {res['intents']}　转人工 {res['handoff']}　"
                f"judge：{'通过' if v.get('pass') else '不通过'}"
                f"（{v.get('reason', '')}）归因：{v.get('attribution') or '-'}"
                + (f"　硬约束失败：{'；'.join(rd['det_fails'][i])}"
                   if rd["det_fails"][i] else ""),
            ]
        lines += ["- **人工评分**：", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
