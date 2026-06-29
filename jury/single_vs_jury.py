"""TASK 1 - the demo number: single-judge baseline vs 5-family jury.

Task type: SUMMARY FAITHFULNESS (binary FAITHFUL / UNFAITHFUL). 15 hand-labeled items,
authored as a fair benchmark (mix of clear and subtle cases), NOT tuned to make the jury
win. Two configs on the SAME set:
  (a) single-model baseline = phi-4-14b  (chosen a-priori, not cherry-picked post hoc)
  (b) 5-family jury, MAJORITY vote (3+/5)
We report agreement-with-gold for each, per-model agreement, and tokens + latency per path.

Honesty: whatever the numbers say, they stand. Majority (not accuracy-weighted) vote is used
because weighting judges by their accuracy on THIS set is circular without a held-out split.

Run: cd ~/projects/my-unsloth-finetune && source env-wsl.sh \
       && jury/.venv/bin/python jury/single_vs_jury.py
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

SCHEMA = {"name": "faithfulness", "strict": True, "schema": {
    "type": "object",
    "properties": {"reason": {"type": "string"},
                   "verdict": {"type": "string", "enum": ["FAITHFUL", "UNFAITHFUL"]}},
    "required": ["reason", "verdict"], "additionalProperties": False}}

RUBRIC = ("You check whether a SUMMARY is faithful to its SOURCE. "
          "FAITHFUL = every claim in the summary is supported by the source (paraphrase and "
          "omission are fine). UNFAITHFUL = the summary adds, changes, or contradicts any fact "
          "not in the source. Reply ONLY with the JSON object.")

# Hand-labeled gold set (7 FAITHFUL, 8 UNFAITHFUL). Subtle items included on purpose.
GOLD = [
    {"id": "s1", "label": "FAITHFUL",
     "source": "The city council approved a new bike lane on Main Street. Construction begins in March and is expected to finish by June.",
     "summary": "The council approved a Main Street bike lane, with construction running from March to June."},
    {"id": "s2", "label": "UNFAITHFUL",
     "source": "The library extended its weekend hours starting this month.",
     "summary": "The library extended its weekend hours this month after a survey of patrons."},
    {"id": "s3", "label": "UNFAITHFUL",
     "source": "The company hired 200 new employees last quarter.",
     "summary": "The company hired 2,000 new employees last quarter."},
    {"id": "s4", "label": "FAITHFUL",
     "source": "Rainfall this year was below average, leading to water restrictions.",
     "summary": "Lower-than-usual rain caused water restrictions this year."},
    {"id": "s5", "label": "UNFAITHFUL",
     "source": "Quarterly profits fell by 5% compared to last year.",
     "summary": "Quarterly profits rose by 5% compared to last year."},
    {"id": "s6", "label": "FAITHFUL",
     "source": "The museum will display the painting until August, after which it returns to a private collection.",
     "summary": "The painting is on display at the museum until August, then goes back to a private owner."},
    {"id": "s7", "label": "UNFAITHFUL",
     "source": "Flight 482 was delayed for three hours yesterday.",
     "summary": "Flight 482 was delayed three hours yesterday due to bad weather."},
    {"id": "s8", "label": "FAITHFUL",
     "source": "The vaccine is recommended for adults over 65 and people with chronic conditions.",
     "summary": "The vaccine is advised for those over 65 and people with chronic illnesses."},
    {"id": "s9", "label": "UNFAITHFUL",
     "source": "Some students reported difficulty accessing the online portal.",
     "summary": "All students reported difficulty accessing the online portal."},
    {"id": "s10", "label": "FAITHFUL",
     "source": "The startup raised $4 million in seed funding to expand its engineering team.",
     "summary": "The startup raised $4M in seed funding to grow its engineering team."},
    {"id": "s11", "label": "UNFAITHFUL",
     "source": "The conference is scheduled for September 12th.",
     "summary": "The conference is scheduled for September 21st."},
    {"id": "s12", "label": "FAITHFUL",
     "source": "The restaurant, which opened in 2019, won a local award for its desserts and now plans a second location.",
     "summary": "The restaurant won a local dessert award and plans to open a second location."},
    {"id": "s13", "label": "UNFAITHFUL",
     "source": "The team released a software update that fixes several bugs.",
     "summary": "The team released a software update that fixes 12 bugs."},
    {"id": "s14", "label": "FAITHFUL",
     "source": "Voter turnout reached a record high in the latest election.",
     "summary": "The latest election saw record-high voter turnout."},
    {"id": "s15", "label": "UNFAITHFUL",
     "source": "The new policy applies only to full-time employees.",
     "summary": "The new policy applies to all employees."},
]


def judge(model, item):
    prompt = f"SOURCE:\n{item['source']}\n\nSUMMARY:\n{item['summary']}"
    t0 = time.time()
    try:
        r = litellm.completion(
            model=model,
            messages=[{"role": "system", "content": RUBRIC}, {"role": "user", "content": prompt}],
            response_format={"type": "json_schema", "json_schema": SCHEMA}, temperature=0, timeout=600)
        v = json.loads(r.choices[0].message.content)["verdict"]
        usage = getattr(r, "usage", None)
        return {"verdict": v, "tokens": getattr(usage, "total_tokens", 0) or 0, "secs": time.time() - t0}
    except Exception as e:  # noqa: BLE001
        return {"verdict": None, "error": str(e)[:120], "tokens": 0, "secs": time.time() - t0}


def main():
    print(f"single-vs-jury | {len(GOLD)} items | baseline={BASELINE.split('/')[-1]} | api={os.getenv('LM_STUDIO_API_BASE')}\n")
    raw = {m: {} for m in JUDGES}
    for m in JUDGES:
        short = m.split("/")[-1]
        print(f"-- judge: {short} --", flush=True)
        for it in GOLD:
            raw[m][it["id"]] = judge(m, it)
        c = sum(1 for it in GOLD if raw[m][it["id"]]["verdict"] == it["label"])
        print(f"   accuracy {c}/{len(GOLD)}", flush=True)

    N = len(GOLD)
    permodel = {}
    for m in JUDGES:
        permodel[m] = {
            "correct": sum(1 for it in GOLD if raw[m][it["id"]]["verdict"] == it["label"]),
            "tokens": sum(raw[m][it["id"]]["tokens"] for it in GOLD),
            "secs": round(sum(raw[m][it["id"]]["secs"] for it in GOLD), 1),
            "errors": sum(1 for it in GOLD if raw[m][it["id"]].get("error")),
        }

    # jury majority vote per item
    jury_correct = 0
    per_item = []
    for it in GOLD:
        votes = [raw[m][it["id"]]["verdict"] for m in JUDGES]
        fa, un = votes.count("FAITHFUL"), votes.count("UNFAITHFUL")
        maj = "FAITHFUL" if fa > un else "UNFAITHFUL"  # tie/None -> UNFAITHFUL (stricter)
        ok = maj == it["label"]
        jury_correct += ok
        per_item.append({"id": it["id"], "gold": it["label"],
                         "single": raw[BASELINE][it["id"]]["verdict"], "jury": maj,
                         "fa": fa, "un": un})
    single_correct = permodel[BASELINE]["correct"]
    single_tok, single_secs = permodel[BASELINE]["tokens"], permodel[BASELINE]["secs"]
    jury_tok = sum(permodel[m]["tokens"] for m in JUDGES)
    jury_secs = round(sum(permodel[m]["secs"] for m in JUDGES), 1)
    best_single = max(permodel.items(), key=lambda kv: kv[1]["correct"])

    # where heterogeneity helped / hurt vs the single baseline
    helped = [r["id"] for r in per_item if r["single"] != r["gold"] and r["jury"] == r["gold"]]
    hurt = [r["id"] for r in per_item if r["single"] == r["gold"] and r["jury"] != r["gold"]]

    def pct(c):
        return f"{c}/{N} ({100*c/N:.0f}%)"

    # ---- write results markdown ----
    lines = [f"# Jury - single judge vs 5-family ensemble ({datetime.now(timezone.utc).date()})", "",
             f"Task: **summary faithfulness** (binary FAITHFUL/UNFAITHFUL), **{N}** hand-labeled items "
             f"(7 FAITHFUL / 8 UNFAITHFUL). Judges run sequentially (12 GB VRAM, one model at a time). "
             f"Honest run - gold set authored as a fair benchmark, not tuned. Majority vote (not "
             f"accuracy-weighted: weighting by accuracy on the same set is circular).", "",
             "## Headline", "",
             f"- **Single judge ({BASELINE.split('/')[-1]})**: agreement {pct(single_correct)}",
             f"- **5-family jury (majority)**: agreement {pct(jury_correct)}",
             f"- **Best single model**: {best_single[0].split('/')[-1]} {pct(best_single[1]['correct'])}",
             f"- Delta jury - single: **{jury_correct - single_correct:+d}** items",
             f"- Jury vs single beyond cost: jury used **{jury_tok} tokens / {jury_secs}s** vs single "
             f"**{single_tok} tokens / {single_secs}s** (~{jury_tok/max(single_tok,1):.1f}x tokens, "
             f"~{jury_secs/max(single_secs,0.1):.1f}x latency).", "",
             "## Per-model agreement (cost is the price of adding this judge to the panel)", "",
             "| Judge | family | agreement | tokens | latency (s) | errors |",
             "|---|---|---|---|---|---|"]
    fam = {"phi-4-14b": "Microsoft", "ministral-3-14b-reasoning": "Mistral",
           "llama-3.3-8b-instruct-i1": "Meta", "gemma-4-e4b-it": "Google",
           "deepseek-r1-distill-qwen-14b": "DeepSeek"}
    for m in JUDGES:
        s = m.split("/")[-1]
        p = permodel[m]
        lines.append(f"| {s} | {fam.get(s,'?')} | {pct(p['correct'])} | {p['tokens']} | {p['secs']} | {p['errors']} |")
    lines += ["", "## Single vs jury, per item (audit trail)", "",
              "| item | gold | single (phi-4) | jury maj | votes F/U |", "|---|---|---|---|---|"]
    for r in per_item:
        lines.append(f"| {r['id']} | {r['gold']} | {r['single']} | {r['jury']} | {r['fa']}/{r['un']} |")
    lines += ["", "## Reading the result", "",
              f"- Heterogeneity HELPED on items where single was wrong but the jury was right: "
              f"{helped or 'none'}.",
              f"- Heterogeneity HURT on items where single was right but the jury was wrong: "
              f"{hurt or 'none'}.",
              "- The jury costs ~5x tokens and ~5x latency (5 models, sequential JIT loads). It is worth "
              "it only when the agreement gain offsets that, i.e. on subtle/ambiguous items where "
              "single-model errors are de-correlated across families. On clear-cut items a single good "
              "judge matches the panel at 1/5 the cost.", ""]
    out = HERE / "results-2026-06-22.md"
    out.write_text("\n".join(lines))

    print("\n=== HEADLINE ===")
    print(f"  single ({BASELINE.split('/')[-1]}): {pct(single_correct)}  | tokens {single_tok} | {single_secs}s")
    print(f"  jury (majority, 5):          {pct(jury_correct)}  | tokens {jury_tok} | {jury_secs}s")
    print(f"  best single: {best_single[0].split('/')[-1]} {pct(best_single[1]['correct'])}")
    print(f"  delta jury-single: {jury_correct - single_correct:+d} items | helped={helped} hurt={hurt}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
