"""Load a Hugging Face dataset into the dashboard as companion training cases.

Uses the public HF datasets-server REST API (no `datasets` lib, no key for public sets),
maps two columns to (input, output), and saves to the dashboard SQLite so it shows up in the
Data / Fine-tune tabs. Tuned for shirshatzman/flirtflip-dataset (original -> playful), but the
columns are parameters so other datasets work too.

Run:  dashboard/.venv/bin/python companion/load_hf.py
      dashboard/.venv/bin/python companion/load_hf.py --dataset X --in-col a --out-col b --target name --max 300
"""
import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
import db  # noqa: E402
from seed_companion import PERSONA  # noqa: E402

API = "https://datasets-server.huggingface.co/rows"


def fetch_rows(dataset, config="default", split="train", max_rows=300):
    rows, off = [], 0
    while len(rows) < max_rows:
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                    "offset": off, "length": 100})
        with urllib.request.urlopen(f"{API}?{q}", timeout=40) as r:
            d = json.load(r)
        batch = [x["row"] for x in d.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        off += len(batch)
        if off >= d.get("num_rows_total", off):
            break
    return rows[:max_rows]


def load(dataset, in_col="original", out_col="playful", target="flirtflip", max_rows=300):
    rows = fetch_rows(dataset, max_rows=max_rows)
    n = 0
    for r in rows:
        inp = str(r.get(in_col) or "").strip()
        out = str(r.get(out_col) or "").strip()
        if inp and out:
            ctx = PERSONA + (f"  [scenario: {r['scenario']}]" if r.get("scenario") else "")
            db.add_case(target, inp, out, ctx, "")
            n += 1
    return n, len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="shirshatzman/flirtflip-dataset")
    ap.add_argument("--in-col", default="original")
    ap.add_argument("--out-col", default="playful")
    ap.add_argument("--target", default="flirtflip")
    ap.add_argument("--max", type=int, default=300)
    a = ap.parse_args()
    db.init()
    n, total = load(a.dataset, a.in_col, a.out_col, a.target, a.max)
    print(f"loaded {n}/{total} rows from {a.dataset} -> dataset '{a.target}' "
          f"(now {len(db.list_cases(a.target))} cases). Visible in the dashboard Data tab.")
