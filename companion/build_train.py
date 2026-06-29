"""Assemble companion_train from the Mira-voice data + a warmth slice, and export the
training JSONL for Colab. Targets the metrics that matter (voice + bake-in), not the mean.

Sources (all framed in Mira's voice via the persona system prompt):
  - companion   : synthetic Mira + hand seed  (the VOICE core)
  - flirtflip    : flirty phrasing/style
  - empathetic   : a slice of Estwld/empathetic_dialogues_llm (warmth, anti-forgetting)
companion_hard is NEVER included - it stays a held-out test for honest before/after.

Bake-in trick: a fraction of the Mira-voice examples is ALSO emitted with NO system prompt,
so the model learns to hold Mira's voice even without the full persona card (targets the
no-prompt metric: base dropped 4.46 -> 2.75 without the prompt).

Run (after the synthetic scale finishes):
  dashboard/.venv/bin/python companion/build_train.py --empathetic 100 --bakein 0.25
Outputs: db dataset 'companion_train' (for the dashboard) + companion/companion_train.jsonl (for Colab).
"""
import argparse
import json
import urllib.parse
import urllib.request
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
import db  # noqa: E402
from seed_companion import PERSONA  # noqa: E402

API = "https://datasets-server.huggingface.co/rows"


def fetch_empathetic(n_pairs):
    """Estwld/empathetic_dialogues_llm -> (user, assistant) pairs from the conversations field."""
    pairs, off = [], 0
    while len(pairs) < n_pairs and off < 2000:
        q = urllib.parse.urlencode({"dataset": "Estwld/empathetic_dialogues_llm", "config": "default",
                                    "split": "train", "offset": off, "length": 100})
        try:
            rows = json.load(urllib.request.urlopen(f"{API}?{q}", timeout=40)).get("rows", [])
        except Exception:
            break
        if not rows:
            break
        for r in rows:
            conv = r.get("row", {}).get("conversations") or []
            for i in range(len(conv) - 1):
                ra = conv[i].get("role") or ("user" if i % 2 == 0 else "assistant")
                rb = conv[i + 1].get("role") or ("user" if (i + 1) % 2 == 0 else "assistant")
                if ra == "user" and rb == "assistant":
                    u = str(conv[i].get("content", "")).strip()
                    a = str(conv[i + 1].get("content", "")).strip()
                    if u and a:
                        pairs.append((u, a))
        off += 100
    return pairs[:n_pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mira-dataset", default="companion", help="the Mira-voice db dataset (v2: companion_v2)")
    ap.add_argument("--flirt", type=int, default=10**6, help="how many flirtflip examples to include (0 = none)")
    ap.add_argument("--empathetic", type=int, default=100)
    ap.add_argument("--bakein", type=float, default=0.25, help="fraction of Mira-voice examples also emitted with no system prompt")
    ap.add_argument("--out", default="companion_train", help="output db dataset + <out>.jsonl name")
    a = ap.parse_args()

    mira = [(c["input"], c["output"]) for c in db.list_cases(a.mira_dataset)]        # voice core
    flirt = [(c["input"], c["output"]) for c in db.list_cases("flirtflip")][:a.flirt]  # style (v2: 0)
    emp = fetch_empathetic(a.empathetic)                                             # warmth slice
    print(f"sources: {a.mira_dataset}(Mira) {len(mira)} | flirtflip {len(flirt)} | empathetic {len(emp)}")

    # rebuild the db dataset <out> (Mira voice for all; for dashboard view/split)
    for old in db.list_cases(a.out):
        db.delete_case(old["id"])
    for u, o in mira + flirt + emp:
        db.add_case(a.out, u, o, PERSONA, "")
    print(f"db dataset '{a.out}': {len(db.list_cases(a.out))} cases")

    # export the training JSONL (with bake-in on the Mira-voice subset)
    out = Path(__file__).resolve().parent / f"{a.out}.jsonl"
    n_full = n_bake = 0
    with out.open("w", encoding="utf-8") as f:
        for src, do_bakein in ((mira, True), (flirt, False), (emp, False)):
            for idx, (u, o) in enumerate(src):
                f.write(json.dumps({"messages": [
                    {"role": "system", "content": PERSONA},
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": o}]}, ensure_ascii=False) + "\n")
                n_full += 1
                # bake-in: emit a copy with NO system prompt for a deterministic fraction
                if do_bakein and a.bakein > 0 and (idx % max(1, round(1 / a.bakein)) == 0):
                    f.write(json.dumps({"messages": [
                        {"role": "user", "content": u},
                        {"role": "assistant", "content": o}]}, ensure_ascii=False) + "\n")
                    n_bake += 1
    print(f"wrote {out.name}: {n_full} examples + {n_bake} bake-in (no-prompt) = {n_full + n_bake} total")
    print("held-out test 'companion_hard' is NOT in here (honest before/after).")


if __name__ == "__main__":
    main()
