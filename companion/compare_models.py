"""See it run: score several models on the SAME cases and print what each said + why.

For each case: every model-under-test generates a reply (using the case context as the system
prompt = the persona), then one judge scores it with a metric. Prints reply snippet + score +
the judge's reason, and a per-model mean. Optionally saves a run per model to the dashboard.

Run:  source env-wsl.sh && dashboard/.venv/bin/python companion/compare_models.py \
        --models llama-3.3-8b-instruct-i1,dolphin3-nsfw-gf-chat --judge gemma-4-e4b-it --n 4
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
import db  # noqa: E402
import evals_lib  # noqa: E402


def derive_base():
    if os.getenv("LM_STUDIO_API_BASE"):
        return os.getenv("LM_STUDIO_API_BASE")
    try:
        gw = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True, timeout=3).stdout.split()[2]
        return f"http://{gw}:1234/v1"
    except Exception:
        return "http://localhost:1234/v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama-3.3-8b-instruct-i1,dolphin3-nsfw-gf-chat")
    ap.add_argument("--judge", default="gemma-4-e4b-it")
    ap.add_argument("--metric", default="Persona-fit (in-character + smart)")
    ap.add_argument("--dataset", default="companion")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--save", action="store_true", help="store a run per model in the dashboard")
    a = ap.parse_args()
    base = derive_base()
    judge_fn = evals_lib.METRICS[a.metric]
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    cases = db.list_cases(a.dataset)[: a.n]
    if not cases:
        print(f"no cases in '{a.dataset}'"); return

    print(f"\nMETRIC: {a.metric}   JUDGE: {a.judge}   cases: {len(cases)}   models: {len(models)}\n")

    # PHASE 1 - generation: each model-under-test is loaded ONCE and answers all cases
    # (no reload per case; the judge isn't touched yet). The fast way on a single GPU.
    print("PHASE 1 - generate (each model loaded once)")
    gens = {}
    for model in models:
        print(f"  {model}: generating {len(cases)} ...")
        out = []
        for c in cases:
            try:
                g = evals_lib.generate(model, base, c["input"], system=c.get("context") or "")
            except Exception as e:  # noqa: BLE001
                g = f"[gen error: {str(e)[:80]}]"
            out.append((c, g))
        gens[model] = out

    # PHASE 2 - judge: the judge is loaded ONCE and scores everything generated above.
    print(f"\nPHASE 2 - judge all with {a.judge} (loaded once)\n")
    means = {}
    for model in models:
        scores = []
        rid = db.add_run(a.dataset, f"{a.metric} @ {model}", model, len(cases), "") if a.save else None
        print(f"===== MODEL UNDER TEST: {model} =====")
        for c, gen in gens[model]:
            try:
                r = judge_fn(a.judge, base, question=c["input"], answer=gen, context=c.get("context") or "")
            except Exception as e:  # noqa: BLE001
                r = {"score": None, "reason": f"[judge error: {str(e)[:80]}]"}
            sc = r.get("score")
            if isinstance(sc, (int, float)):
                scores.append(sc)
            if a.save and rid:
                db.add_result(rid, c["id"], sc, "", f"GEN: {gen[:200]}\nJUDGE: {r.get('reason','')}", 0, 0)
            snippet = " ".join(gen.split())[:90]
            print(f"  U: {c['input'][:48]!r}")
            print(f"     -> {snippet!r}")
            print(f"     score {sc}/5 | {(' '.join(str(r.get('reason','')).split()))[:100]}")
        mean = sum(scores) / len(scores) if scores else None
        means[model] = mean
        if a.save and rid:
            with db.conn() as cc:
                cc.execute("UPDATE runs SET summary=? WHERE id=?", (f"mean {mean:.2f}" if mean else "-", rid))
        print(f"  >>> {model}: mean {mean:.2f}/5\n" if mean is not None else f"  >>> {model}: no scores\n")

    print("SUMMARY (mean, higher = more in-character + tasteful + smart):")
    for m, v in sorted(means.items(), key=lambda kv: -(kv[1] or 0)):
        print(f"  {v:.2f}  {m}" if v is not None else f"  -     {m}")
    if a.save:
        print("\n(saved as runs in the dashboard -> Results tab)")


if __name__ == "__main__":
    main()
