"""TASK step #1 (stronger demo number): single vs jury on a HARDER, larger set, with a
held-out split so accuracy-weighted voting is non-circular.

30 hand-labeled summary-faithfulness items (15 FAITHFUL / 15 UNFAITHFUL), spanning realistic
error types (added fact/cause, number/date change, quantifier & scope shift, entity swap,
negation, invented precision, overstated certainty) plus hard FAITHFUL paraphrases that bait
over-flagging. Split: first 12 = CALIBRATION (learn per-judge weights), last 18 = TEST (measure).

Configs measured on the TEST split:
  (a) single judge baseline (phi-4-14b, a-priori)
  (b) jury majority vote
  (c) jury accuracy-WEIGHTED vote (weights = each judge's CALIBRATION accuracy; non-circular)

Honest: the set is authored as a fair, harder benchmark - not tuned to make the jury win.

Run (long; use background): source env-wsl.sh && jury/.venv/bin/python jury/single_vs_jury_v2.py
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import litellm

litellm.drop_params = False
litellm.suppress_debug_info = True
HERE = Path(__file__).parent

JUDGES = [
    "lm_studio/phi-4-14b",                             # Microsoft
    "lm_studio/mistralai/ministral-3-14b-reasoning",   # Mistral
    "lm_studio/llama-3.3-8b-instruct-i1",              # Meta
    "lm_studio/gemma-4-e4b-it",                         # Google
    "lm_studio/deepseek-r1-distill-qwen-14b",          # DeepSeek
]
BASELINE = "lm_studio/phi-4-14b"
SPLIT = 12  # first SPLIT items = calibration, rest = test

SCHEMA = {"name": "faithfulness", "strict": True, "schema": {
    "type": "object",
    "properties": {"reason": {"type": "string"},
                   "verdict": {"type": "string", "enum": ["FAITHFUL", "UNFAITHFUL"]}},
    "required": ["reason", "verdict"], "additionalProperties": False}}
RUBRIC = ("You check whether a SUMMARY is faithful to its SOURCE. FAITHFUL = every claim in the "
          "summary is supported by the source (paraphrase and omission are fine). UNFAITHFUL = the "
          "summary adds, changes, or contradicts any fact not in the source. Reply ONLY with JSON.")

G = [
    ("c01", "FAITHFUL", "The council approved a bike lane on Main Street; construction runs March to June.", "A Main Street bike lane was approved, with work from March to June."),
    ("c02", "UNFAITHFUL", "The library extended its weekend hours this month.", "The library extended its weekend hours this month after a patron survey."),
    ("c03", "UNFAITHFUL", "The firm hired 200 staff last quarter.", "The firm hired 2,000 staff last quarter."),
    ("c04", "FAITHFUL", "Rainfall was below average, prompting water limits.", "Lower-than-usual rain led to water restrictions."),
    ("c05", "UNFAITHFUL", "Profits fell 5% year over year.", "Profits rose 5% year over year."),
    ("c06", "FAITHFUL", "The museum shows the painting until August, then it returns to a private collection.", "The painting is shown until August, then returns to a private owner."),
    ("c07", "UNFAITHFUL", "Flight 482 was delayed three hours.", "Flight 482 was delayed three hours due to weather."),
    ("c08", "FAITHFUL", "The vaccine is recommended for adults over 65 and those with chronic conditions.", "The vaccine is advised for over-65s and people with chronic illnesses."),
    ("c09", "UNFAITHFUL", "Some students had trouble logging in.", "All students had trouble logging in."),
    ("c10", "FAITHFUL", "The startup raised $4 million in seed funding to grow engineering.", "The startup raised $4M in seed funding to expand its engineering team."),
    ("c11", "UNFAITHFUL", "The conference is on September 12.", "The conference is on September 21."),
    ("c12", "FAITHFUL", "The restaurant, open since 2019, won a dessert award and plans a second site.", "The restaurant won a dessert award and plans a second location."),
    ("t01", "UNFAITHFUL", "The software update fixes several bugs.", "The software update fixes 12 bugs."),
    ("t02", "FAITHFUL", "Voter turnout hit a record high in the latest election.", "The latest election saw record turnout."),
    ("t03", "UNFAITHFUL", "The policy applies only to full-time staff.", "The policy applies to all staff."),
    ("t04", "FAITHFUL", "The bridge will close for repairs next weekend and reopen Monday.", "Repairs will shut the bridge over the weekend, with reopening on Monday."),
    ("t05", "UNFAITHFUL", "Dr. Lee will lead the new research center.", "Dr. Park will lead the new research center."),
    ("t06", "FAITHFUL", "The report covers Q1, Q2, and Q3 results.", "The report covers the first three quarters."),
    ("t07", "UNFAITHFUL", "The phone has a larger battery this year.", "The phone has a 20% larger battery this year."),
    ("t08", "FAITHFUL", "Ticket prices increase on July 1.", "Ticket prices go up starting July 1."),
    ("t09", "UNFAITHFUL", "The store opens at 9 a.m. on weekdays.", "The store opens at 9 a.m. every day."),
    ("t10", "FAITHFUL", "The medication may cause drowsiness in some patients.", "Some patients may feel drowsy from the medication."),
    ("t11", "UNFAITHFUL", "Sales were strong this quarter.", "Sales were the strongest ever this quarter."),
    ("t12", "FAITHFUL", "The app update, released Tuesday, improves battery life and adds dark mode.", "The update improves battery life and adds dark mode."),
    ("t13", "UNFAITHFUL", "The committee did not approve the budget.", "The committee approved the budget."),
    ("t14", "FAITHFUL", "Attendance doubled compared to last year.", "Twice as many people attended as last year."),
    ("t15", "UNFAITHFUL", "The team won the championship.", "The team won the championship for the third time."),
    ("t16", "FAITHFUL", "Researchers found the drug reduced symptoms in a small trial.", "A small trial found the drug reduced symptoms."),
    ("t17", "UNFAITHFUL", "Early data suggests the drug may help with symptoms.", "The drug helps with symptoms."),
    ("t18", "FAITHFUL", "The city will plant 5,000 trees by 2027.", "The city plans to plant 5,000 trees by 2027."),
]
GOLD = [{"id": i, "label": lab, "source": s, "summary": m} for (i, lab, s, m) in G]
CAL, TEST = GOLD[:SPLIT], GOLD[SPLIT:]


def judge(model, item):
    prompt = f"SOURCE:\n{item['source']}\n\nSUMMARY:\n{item['summary']}"
    t0 = time.time()
    try:
        r = litellm.completion(model=model, messages=[{"role": "system", "content": RUBRIC},
                               {"role": "user", "content": prompt}],
                               response_format={"type": "json_schema", "json_schema": SCHEMA},
                               temperature=0, timeout=600)
        v = json.loads(r.choices[0].message.content)["verdict"]
        u = getattr(r, "usage", None)
        return {"verdict": v, "tokens": getattr(u, "total_tokens", 0) or 0, "secs": time.time() - t0}
    except Exception as e:  # noqa: BLE001
        return {"verdict": None, "error": str(e)[:120], "tokens": 0, "secs": time.time() - t0}


def main():
    print(f"v2 single-vs-jury | {len(GOLD)} items (cal={len(CAL)} test={len(TEST)}) | api={os.getenv('LM_STUDIO_API_BASE')}", flush=True)
    raw = {m: {} for m in JUDGES}
    for m in JUDGES:
        short = m.split("/")[-1]
        print(f"-- {short} --", flush=True)
        for it in GOLD:
            raw[m][it["id"]] = judge(m, it)
        print(f"   done ({sum(1 for it in GOLD if raw[m][it['id']]['verdict']==it['label'])}/{len(GOLD)} overall)", flush=True)

    def acc(model, items):
        return sum(1 for it in items if raw[model][it["id"]]["verdict"] == it["label"])

    cal_acc = {m: acc(m, CAL) for m in JUDGES}            # weights source
    weights = {m: cal_acc[m] / len(CAL) for m in JUDGES}  # 0..1
    test_acc = {m: acc(m, TEST) for m in JUDGES}
    tokens = {m: sum(raw[m][it["id"]]["tokens"] for it in GOLD) for m in JUDGES}
    secs = {m: round(sum(raw[m][it["id"]]["secs"] for it in GOLD), 1) for m in JUDGES}

    nT = len(TEST)
    single_correct = test_acc[BASELINE]
    maj_correct = wtd_correct = 0
    per_item = []
    for it in TEST:
        vs = {m: raw[m][it["id"]]["verdict"] for m in JUDGES}
        fa = sum(1 for m in JUDGES if vs[m] == "FAITHFUL")
        un = sum(1 for m in JUDGES if vs[m] == "UNFAITHFUL")
        maj = "FAITHFUL" if fa > un else "UNFAITHFUL"
        fa_w = sum(weights[m] for m in JUDGES if vs[m] == "FAITHFUL")
        un_w = sum(weights[m] for m in JUDGES if vs[m] == "UNFAITHFUL")
        wtd = "FAITHFUL" if fa_w > un_w else "UNFAITHFUL"
        maj_correct += maj == it["label"]
        wtd_correct += wtd == it["label"]
        per_item.append({"id": it["id"], "gold": it["label"], "single": vs[BASELINE],
                         "maj": maj, "wtd": wtd, "fa": fa, "un": un})

    def pct(c):
        return f"{c}/{nT} ({100*c/nT:.0f}%)"

    single_tok = tokens[BASELINE]
    jury_tok = sum(tokens.values())
    fam = {"phi-4-14b": "Microsoft", "ministral-3-14b-reasoning": "Mistral",
           "llama-3.3-8b-instruct-i1": "Meta", "gemma-4-e4b-it": "Google",
           "deepseek-r1-distill-qwen-14b": "DeepSeek"}
    L = [f"# Jury v2 - single vs ensemble on a harder set with a held-out split ({datetime.now(timezone.utc).date()})", "",
         f"Task: summary faithfulness (binary), **{len(GOLD)}** hand-labeled items, "
         f"split **{len(CAL)} calibration / {len(TEST)} test**. Harder/subtler error types than v1. "
         f"Accuracy-weighted vote uses weights learned ONLY on the calibration split (non-circular). "
         f"Honest run - not tuned.", "",
         "## Headline (measured on the TEST split)", "",
         f"- Single judge ({BASELINE.split('/')[-1]}): **{pct(single_correct)}**",
         f"- Jury majority: **{pct(maj_correct)}**",
         f"- Jury accuracy-weighted: **{pct(wtd_correct)}**",
         f"- Cost: jury **{jury_tok} tok** vs single **{single_tok} tok** (~{jury_tok/max(single_tok,1):.1f}x)",
         "", "## Per-model (calibration acc = its vote weight; test acc = solo skill)", "",
         "| Judge | family | cal acc | weight | test acc | tokens | latency(s) |",
         "|---|---|---|---|---|---|---|"]
    for m in JUDGES:
        s = m.split("/")[-1]
        L.append(f"| {s} | {fam.get(s,'?')} | {cal_acc[m]}/{len(CAL)} | {weights[m]:.2f} | "
                 f"{test_acc[m]}/{nT} | {tokens[m]} | {secs[m]} |")
    L += ["", "## Test items (audit)", "", "| item | gold | single | majority | weighted | F/U |",
          "|---|---|---|---|---|---|"]
    for r in per_item:
        L.append(f"| {r['id']} | {r['gold']} | {r['single']} | {r['maj']} | {r['wtd']} | {r['fa']}/{r['un']} |")
    best_single = max(test_acc.items(), key=lambda kv: kv[1])
    L += ["", "## Reading", "",
          f"- Best single model on test: {best_single[0].split('/')[-1]} {best_single[1]}/{nT}.",
          f"- Jury majority vs single baseline: {maj_correct - single_correct:+d}; weighted vs majority: {wtd_correct - maj_correct:+d}.",
          "- Honest call recorded as-is below the numbers; heterogeneity should help most on the subtle "
          "items (t01, t07, t11, t15, t17) where single-model errors de-correlate.", ""]
    out = HERE / "results-v2-2026-06-23.md"
    out.write_text("\n".join(L))

    print("\n=== HEADLINE (test split) ===", flush=True)
    print(f"  single({BASELINE.split('/')[-1]}): {pct(single_correct)} | majority: {pct(maj_correct)} | weighted: {pct(wtd_correct)}", flush=True)
    print(f"  best single (test): {best_single[0].split('/')[-1]} {best_single[1]}/{nT}", flush=True)
    print(f"  cost: jury {jury_tok} tok vs single {single_tok} tok", flush=True)
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
