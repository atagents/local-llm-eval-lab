"""Regression CI gate for the FreeStack trajectory eval – engine #9.

WHAT: Loads gold_traces.json, runs check_trace() over all traces (no LLM, no
network, $0), computes label_verdict_agreement and objective pass-rate, then
diffs against a pinned baseline.json. ZERO tolerance on the objective leg:
label_agreement < baseline triggers exit(1).

WHY: Catch regressions in check_trajectory.py, policy_fsm.py, or the engine
before they quietly invalidate the eval harness.

HOW TO RUN:
  # normal gate (offline, default):
  dashboard/.venv/bin/python evals/ci-gate/gate.py

  # pin a new baseline from the current clean state:
  dashboard/.venv/bin/python evals/ci-gate/gate.py --update-baseline

  # smoke-test the gate logic itself (no LM Studio needed):
  dashboard/.venv/bin/python evals/ci-gate/gate.py --selftest

  # advisory jury leg (needs LM Studio + env-wsl.sh sourced):
  dashboard/.venv/bin/python evals/ci-gate/gate.py --jury
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]   # my-unsloth-finetune root
_TRAJ = _REPO / "evals" / "trajectory-eval"

# Add trajectory-eval to path so we can import its modules.
sys.path.insert(0, str(_TRAJ))
sys.path.insert(0, str(_REPO / "dashboard"))

from check_trajectory import check_trace  # noqa: E402
from trace_schema import load_traces       # noqa: E402

GOLD_TRACES = _TRAJ / "gold_traces.json"
BASELINE_FILE = _HERE / "baseline.json"
DB_FILE = _HERE / "gate.db"
SUMMARY_FILE = _HERE / "summary.md"

_EPSILON = 1e-9   # float safety only – practically zero tolerance


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------
def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_FILE)
    c.row_factory = sqlite3.Row
    return c


def _init_db() -> None:
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS runs(
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT    NOT NULL,
            git_commit  TEXT    NOT NULL,
            label_agreement REAL NOT NULL,
            obj_pass_rate   REAL NOT NULL,
            n           INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trace_results(
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      INTEGER NOT NULL REFERENCES runs(id),
            trace_id    TEXT    NOT NULL,
            label       TEXT    NOT NULL,
            verdict     TEXT    NOT NULL,
            agrees      INTEGER NOT NULL,
            n_illegal   INTEGER NOT NULL,
            legal_rate  REAL
        );
        """)


def _persist_run(commit: str, label_agreement: float, obj_pass_rate: float,
                 per: list[dict]) -> int:
    _init_db()
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO runs(created_at,git_commit,label_agreement,obj_pass_rate,n) "
            "VALUES(?,?,?,?,?)",
            (now, commit, label_agreement, obj_pass_rate, len(per)))
        run_id = cur.lastrowid
        c.executemany(
            "INSERT INTO trace_results"
            "(run_id,trace_id,label,verdict,agrees,n_illegal,legal_rate)"
            " VALUES(?,?,?,?,?,?,?)",
            [(run_id, r["id"], r["label"], r["verdict"],
              int(r["agrees_with_label"]), r["n_illegal"],
              r["legal_transition_rate"]) for r in per])
    return run_id


# ---------------------------------------------------------------------------
# Git commit SHA
# ---------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def _run_checks(traces_path: Path) -> tuple[float, float, list[dict]]:
    """Returns (label_agreement, obj_pass_rate, per_trace_list)."""
    traces = load_traces(traces_path)
    per = [check_trace(t) for t in traces]
    n = len(per)
    if n == 0:
        raise ValueError("no traces found in " + str(traces_path))
    n_pass = sum(1 for r in per if r["verdict"] == "PASS")
    agreed = sum(1 for r in per if r["agrees_with_label"])
    return round(agreed / n, 6), round(n_pass / n, 6), per


# ---------------------------------------------------------------------------
# Markdown summary builder
# ---------------------------------------------------------------------------
def _build_summary(
    commit: str,
    label_agreement: float,
    obj_pass_rate: float,
    baseline: dict,
    per: list[dict],
    verdict: str,           # "PASS" | "REGRESSION"
    new_failing: list[str],
) -> str:
    delta_la = label_agreement - baseline["label_agreement"]
    delta_pr = obj_pass_rate - baseline["obj_pass_rate"]
    sign = lambda v: ("+" if v >= 0 else "") + f"{v:.4f}"

    lines = [
        "## Regression CI gate – objective trajectory check",
        "",
        f"**Commit:** `{commit or 'unknown'}`  "
        f"**Status:** {'PASS' if verdict == 'PASS' else 'REGRESSION DETECTED'}",
        "",
        "| metric | baseline | current | delta |",
        "|---|---|---|---|",
        f"| label_verdict_agreement | {baseline['label_agreement']:.4f}"
        f" | {label_agreement:.4f} | {sign(delta_la)} |",
        f"| obj_pass_rate | {baseline['obj_pass_rate']:.4f}"
        f" | {obj_pass_rate:.4f} | {sign(delta_pr)} |",
        "",
    ]
    if new_failing:
        lines += ["**Newly failing traces:**", ""]
        for tid in new_failing:
            lines.append(f"- `{tid}`")
        lines.append("")
    # per-trace table
    lines += [
        "### Per-trace results",
        "",
        "| id | label | verdict | agrees | illegal |",
        "|---|---|---|---|---|",
    ]
    for r in per:
        lines.append(
            f"| `{r['id']}` | {r['label']} | {r['verdict']}"
            f" | {'yes' if r['agrees_with_label'] else 'NO'}"
            f" | {r['n_illegal']} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Optional advisory jury leg (--jury flag only)
# ---------------------------------------------------------------------------
def _run_jury(per: list[dict], baseline: dict | None) -> None:
    """Advisory only – never changes exit code."""
    import os
    import statistics as _st
    try:
        import litellm  # noqa: F401
    except ImportError:
        print("[jury] litellm not available – skipping advisory jury leg")
        return

    sys.path.insert(0, str(_TRAJ))
    from score_soft import judge, FAST_JURY  # noqa: E402

    api_base = os.getenv("LM_STUDIO_API_BASE", "http://localhost:1234/v1")
    traces = [t for t in load_traces(GOLD_TRACES) if t.transcript]
    if not traces:
        print("[jury] no traces with transcript – skipping")
        return
    id_set = {r["id"] for r in per}
    traces = [t for t in traces if t.id in id_set]
    print(f"\n[jury] ADVISORY – {len(FAST_JURY)} judges x {len(traces)} traces (needs LM Studio)")

    soft: dict[str, list] = {t.id: [] for t in traces}
    for m in FAST_JURY:
        for t in traces:
            try:
                d = judge(m, api_base, t.transcript)
                soft[t.id].append(d["score"])
            except Exception as e:  # noqa: BLE001
                print(f"[jury] {m}/{t.id}: {e}")

    scores = [_st.median(v) for v in soft.values() if v]
    if not scores:
        print("[jury] all jury calls failed – no score")
        return

    jury_median = round(_st.median(scores), 3)
    print(f"[jury] jury_median = {jury_median:.3f} (scale 1–5)")

    if baseline and "jury_median" in baseline:
        print(f"[jury] baseline jury_median = {baseline['jury_median']:.3f}  "
              f"delta = {jury_median - baseline['jury_median']:+.3f}")
    else:
        print("[jury] no baseline jury_median – reporting raw only")
    print("[jury] verdict is ADVISORY only – exit code unchanged")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Regression CI gate – trajectory eval")
    ap.add_argument("--update-baseline", action="store_true",
                    help="write current run as new baseline.json and exit 0")
    ap.add_argument("--selftest", action="store_true",
                    help="smoke-test the gate logic; seeded regression must yield exit-1, "
                         "clean run must yield exit-0; prints PASS; no LM Studio needed")
    ap.add_argument("--jury", action="store_true",
                    help="also run the advisory soft jury (needs LM Studio running)")
    return ap.parse_args()


def _gate(traces_path: Path, baseline: dict | None,
          update_baseline: bool, commit: str) -> int:
    """Run the hard gate. Returns exit code (0 = pass, 1 = regression)."""
    label_agreement, obj_pass_rate, per = _run_checks(traces_path)

    if update_baseline:
        payload = {"label_agreement": label_agreement, "obj_pass_rate": obj_pass_rate,
                   "updated_at": datetime.now(timezone.utc).isoformat(), "commit": commit}
        BASELINE_FILE.write_text(json.dumps(payload, indent=2))
        print(f"baseline.json updated: label_agreement={label_agreement:.4f}  "
              f"obj_pass_rate={obj_pass_rate:.4f}")
        return 0

    if baseline is None:
        print("ERROR: baseline.json not found. Run --update-baseline first.", file=sys.stderr)
        return 1

    # newly failing = any trace that doesn't agree with its gold label now
    new_failing = [r["id"] for r in per
                   if not r["agrees_with_label"] or r["verdict"] == "FAIL"]

    is_regression = label_agreement < baseline["label_agreement"] - _EPSILON
    verdict = "REGRESSION" if is_regression else "PASS"

    # persist to DB (gate.db is gitignored via *.db)
    _persist_run(commit, label_agreement, obj_pass_rate, per)

    # markdown summary – always written; CI workflow posts it as a PR comment on failure
    summary_md = _build_summary(commit, label_agreement, obj_pass_rate,
                                baseline, per, verdict,
                                new_failing if is_regression else [])
    SUMMARY_FILE.write_text(summary_md)

    print(summary_md)

    if is_regression:
        print(f"REGRESSION: label_agreement dropped {label_agreement:.4f} < "
              f"baseline {baseline['label_agreement']:.4f}", file=sys.stderr)
        return 1
    print(f"PASS: label_agreement={label_agreement:.4f} >= baseline "
          f"{baseline['label_agreement']:.4f}")
    return 0


def _selftest() -> None:
    """Seed a regression in-process; assert exit-1 logic; restore; assert exit-0.
    No real files are modified. Prints PASS on success, raises on failure."""
    import os
    import tempfile

    print("selftest: seeding a regression (no real files touched) ...")

    # Load gold traces as raw dicts so we can corrupt one label.
    # Flip the first 'bad' trace label to 'good' so the checker (which returns
    # FAIL for it) will disagree -> label_agreement drops below 1.0.
    raw: list[dict] = json.loads(GOLD_TRACES.read_text())
    corrupted: list[dict] = []
    flipped = False
    for d in raw:
        copy = dict(d)
        if not flipped and copy.get("label") == "bad":
            copy["label"] = "good"
            flipped = True
        corrupted.append(copy)

    assert flipped, "no 'bad' trace found in gold_traces.json to corrupt"

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(json.dumps(corrupted))
        tmp = f.name

    # Fake baseline: perfect agreement, so any degradation triggers regression.
    fake_baseline = {"label_agreement": 1.0, "obj_pass_rate": 0.5}

    try:
        # Seeded regression must return exit-code 1.
        code = _gate(Path(tmp), fake_baseline, update_baseline=False, commit="selftest")
        assert code == 1, f"expected exit-1 for seeded regression, got {code}"
        print("selftest: seeded regression correctly returned exit-1 [OK]")

        # Clean gold traces against the same fake baseline must return exit-code 0.
        code = _gate(GOLD_TRACES, fake_baseline, update_baseline=False, commit="selftest")
        assert code == 0, f"expected exit-0 for clean gold traces, got {code}"
        print("selftest: clean gold traces correctly returned exit-0 [OK]")
    finally:
        os.unlink(tmp)

    print("\nselftest: PASS")


def main() -> None:
    args = _parse()

    if args.selftest:
        _selftest()
        sys.exit(0)

    commit = _git_sha()
    baseline: dict | None = None
    if BASELINE_FILE.exists():
        baseline = json.loads(BASELINE_FILE.read_text())

    exit_code = _gate(GOLD_TRACES, baseline, update_baseline=args.update_baseline,
                      commit=commit)

    if args.jury and not args.update_baseline:
        _, _, per = _run_checks(GOLD_TRACES)
        _run_jury(per, baseline)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
