"""SQLite storage for the eval dashboard. One file, zero config."""
import sqlite3
import time
from pathlib import Path

DB = Path(__file__).parent / "evals.db"


def conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init():
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS cases(
          id INTEGER PRIMARY KEY AUTOINCREMENT, dataset TEXT NOT NULL,
          input TEXT, output TEXT, context TEXT, gold TEXT, created_at REAL);
        CREATE TABLE IF NOT EXISTS runs(
          id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL, dataset TEXT,
          eval_type TEXT, model TEXT, n INTEGER, summary TEXT);
        CREATE TABLE IF NOT EXISTS results(
          id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, case_id INTEGER,
          score REAL, verdict TEXT, reason TEXT, tokens INTEGER, secs REAL);
        """)


# ---- cases ----
def add_case(dataset, inp, out, ctx="", gold=""):
    with conn() as c:
        c.execute("INSERT INTO cases(dataset,input,output,context,gold,created_at) VALUES(?,?,?,?,?,?)",
                  (dataset, inp, out, ctx, gold, time.time()))


def list_cases(dataset=None):
    with conn() as c:
        if dataset:
            return [dict(r) for r in c.execute("SELECT * FROM cases WHERE dataset=? ORDER BY id", (dataset,))]
        return [dict(r) for r in c.execute("SELECT * FROM cases ORDER BY id")]


def delete_case(cid):
    with conn() as c:
        c.execute("DELETE FROM cases WHERE id=?", (cid,))


def datasets():
    with conn() as c:
        return [r[0] for r in c.execute("SELECT DISTINCT dataset FROM cases ORDER BY dataset")]


# ---- runs / results ----
def add_run(dataset, eval_type, model, n, summary):
    with conn() as c:
        cur = c.execute("INSERT INTO runs(created_at,dataset,eval_type,model,n,summary) VALUES(?,?,?,?,?,?)",
                        (time.time(), dataset, eval_type, model, n, summary))
        return cur.lastrowid


def add_result(run_id, case_id, score, verdict, reason, tokens, secs):
    with conn() as c:
        c.execute("INSERT INTO results(run_id,case_id,score,verdict,reason,tokens,secs) VALUES(?,?,?,?,?,?,?)",
                  (run_id, case_id, score, verdict, reason, tokens, secs))


def list_runs():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM runs ORDER BY id DESC")]


def get_results(run_id):
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT r.*, c.input, c.output, c.gold FROM results r "
            "LEFT JOIN cases c ON c.id=r.case_id WHERE r.run_id=? ORDER BY r.id", (run_id,))]
