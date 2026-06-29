"""Dataset train/test split + train-set export for the eval-driven fine-tune loop.

Split is deterministic (hash of case id), so re-running is stable. The split is materialized
as two real datasets `<ds>__train` / `<ds>__test` so the rest of the dashboard (Run, Compare)
can target them directly and you can SEE the split in every dropdown.
"""
import hashlib
import io
import json


def _bucket(case_id, seed):
    """Deterministic 0..1 from a case id (stable across reruns)."""
    h = hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def is_split(name):
    return name.endswith("__train") or name.endswith("__test")


def split(db, source_ds, test_frac=0.2, seed="ft1"):
    """(Re)create <ds>__train / <ds>__test from source_ds. Returns (train_ds, test_ds, n_train, n_test)."""
    train_ds, test_ds = f"{source_ds}__train", f"{source_ds}__test"
    for d in (train_ds, test_ds):                       # wipe any prior split first
        for c in db.list_cases(d):
            db.delete_case(c["id"])
    n_tr = n_te = 0
    for c in db.list_cases(source_ds):
        to_test = _bucket(c["id"], seed) < test_frac
        dst = test_ds if to_test else train_ds
        db.add_case(dst, c["input"], c["output"], c.get("context") or "", c.get("gold") or "")
        n_te += to_test
        n_tr += not to_test
    return train_ds, test_ds, n_tr, n_te


def train_jsonl(db, train_ds):
    """Train cases -> chat JSONL (one {"messages":[...]} per line) for Unsloth/SFT on Colab.
    input -> user turn, output -> assistant turn, context -> system turn (if present)."""
    buf = io.StringIO()
    for c in db.list_cases(train_ds):
        msgs = []
        if c.get("context"):
            msgs.append({"role": "system", "content": c["context"]})
        msgs.append({"role": "user", "content": c["input"]})
        msgs.append({"role": "assistant", "content": c["output"]})
        buf.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    return buf.getvalue().encode("utf-8")
