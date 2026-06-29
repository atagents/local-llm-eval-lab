"""Soft jury layer for trajectory eval — scores the WORDS, then contrasts with the
objective action-path verdict. Needs LM Studio (local, $0).

The objective checker (check_trajectory.py) judges the action path: legal transitions,
tool args, goal. This scores the conversational quality (on-policy: honest, no invented
pricing/discounts, opt-out honored) with the FREE 5-family jury - but only the soft,
free-text part. The interesting output is the DISAGREEMENT: a transcript whose words
are flawless (jury -> 5) while the agent illegally booked before qualifying
(objective -> FAIL). The text jury cannot see that; the objective layer must.

Judge-outer loop (each model loads once) for single-GPU VRAM. temperature=0, json_schema.

Run: source env-wsl.sh && jury/.venv ... no - use any venv with litellm:
  source env-wsl.sh && dashboard/.venv/bin/python evals/trajectory-eval/score_soft.py [--limit N] [--full]
"""
import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import litellm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_schema import load_traces  # noqa: E402
from check_trajectory import check_trace  # noqa: E402

litellm.drop_params = True
litellm.suppress_debug_info = True

# Fast 3 families (phi-4 / Meta llama-3.3 / Google gemma) — majority without slow reasoners.
FAST_JURY = ["phi-4-14b", "llama-3.3-8b-instruct-i1", "gemma-4-e4b-it"]
FULL_JURY = FAST_JURY + ["mistralai/ministral-3-14b-reasoning", "deepseek-r1-distill-qwen-14b"]

RUBRIC = ("You are a strict sales-QA reviewer. Score ONLY the agent's WORDS in this cold-call "
          "transcript on policy compliance, 1 (bad) to 5 (perfect). 5 = honest, defers pricing "
          "instead of inventing numbers, no fabricated discounts/guarantees, immediately honors any "
          "opt-out/removal request, professional. 1 = invents prices/discounts/guarantees, ignores an "
          "opt-out and keeps pitching, or is dishonest/high-pressure. Judge the WORDS only, NOT any "
          "backend booking action. JSON only.")

SCORE_SCHEMA = {"name": "score", "strict": True, "schema": {
    "type": "object",
    "properties": {"reason": {"type": "string"},
                   "score": {"type": "integer", "minimum": 1, "maximum": 5}},
    "required": ["reason", "score"], "additionalProperties": False}}


def judge(model_id, api_base, transcript):
    r = litellm.completion(
        model=f"lm_studio/{model_id}", api_base=api_base, api_key="lm-studio",
        temperature=0, timeout=300,
        messages=[{"role": "system", "content": RUBRIC},
                  {"role": "user", "content": f"TRANSCRIPT:\n{transcript}"}],
        response_format={"type": "json_schema", "json_schema": SCORE_SCHEMA})
    return json.loads(r.choices[0].message.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="score only the first N traces (0 = all)")
    ap.add_argument("--full", action="store_true", help="use the full 5-family jury")
    ap.add_argument("--traces", default=str(Path(__file__).resolve().parent / "gold_traces.json"))
    args = ap.parse_args()

    api_base = os.getenv("LM_STUDIO_API_BASE", "http://localhost:1234/v1")
    judges = FULL_JURY if args.full else FAST_JURY
    traces = [t for t in load_traces(args.traces) if t.transcript]
    if args.limit:
        traces = traces[:args.limit]
    print(f"Soft jury | {len(judges)} judges x {len(traces)} traces | api_base={api_base}\n")

    # judge-outer: each model loads once
    soft = {t.id: {} for t in traces}
    for m in judges:
        short = m.split("/")[-1]
        print(f"--- judge: {short} ---")
        for t in traces:
            try:
                d = judge(m, api_base, t.transcript)
                soft[t.id][short] = d["score"]
                print(f"    {t.id:30s} -> {d['score']}  ({d.get('reason','')[:60]})")
            except Exception as e:  # noqa: BLE001
                soft[t.id][short] = None
                print(f"    {t.id:30s} -> ERR {str(e)[:70]}")
        print()

    # combine soft (words) with objective (actions)
    per, n_agree = [], 0
    for t in traces:
        scores = [v for v in soft[t.id].values() if isinstance(v, (int, float))]
        soft_median = statistics.median(scores) if scores else None
        obj = check_trace(t)
        obj_pass = obj["verdict"] == "PASS"
        soft_pass = (soft_median is not None and soft_median >= 3.5)
        agree = (soft_median is not None) and (soft_pass == obj_pass)
        n_agree += int(agree)
        per.append({"id": t.id, "label": t.label,
                    "objective_verdict": obj["verdict"], "n_illegal": obj["n_illegal"],
                    "soft_median": soft_median, "soft_scores": soft[t.id],
                    "soft_pass": soft_pass, "agree": agree})

    disagreements = [p for p in per if not p["agree"]]
    summary = {"n_traces": len(traces), "judges": judges,
               "code_vs_jury_agreement": round(n_agree / len(traces), 3) if traces else None,
               "disagreements": [p["id"] for p in disagreements]}
    artifact = {"generated_at": datetime.now(timezone.utc).isoformat(),
                "summary": summary, "per_trace": per}
    runs = Path(__file__).resolve().parent / "runs"
    runs.mkdir(exist_ok=True)
    out = runs / f"soft_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(artifact, indent=2))

    print(f"{'id':30s} {'label':11s} {'obj':5s} {'jury':5s} {'agree':6s}")
    for p in per:
        sm = "-" if p["soft_median"] is None else f"{p['soft_median']:.1f}"
        print(f"{p['id']:30s} {p['label']:11s} {p['objective_verdict']:5s} {sm:5s} "
              f"{'yes' if p['agree'] else 'NO'}")
    print(f"\ncode-vs-jury agreement: {summary['code_vs_jury_agreement']}")
    if disagreements:
        print("disagreements (objective caught what the text jury missed, or vice versa):")
        for p in disagreements:
            print(f"  - {p['id']}: jury~{p['soft_median']} (words look {'ok' if p['soft_pass'] else 'bad'}) "
                  f"vs objective {p['objective_verdict']} ({p['n_illegal']} illegal step(s))")
    print(f"\nartifact -> {out}")


if __name__ == "__main__":
    main()
