"""Local LLM Eval Lab - a free, local, private eval dashboard (Streamlit).
Insert test cases, run evals against LM Studio models, see scores + reasoning + cost. No cloud.

Run:  source ../env-wsl.sh && dashboard/.venv/bin/streamlit run dashboard/app.py
Temporary public link (demo):  cloudflared tunnel --url http://localhost:8501
"""
import datetime
import hmac
import json
import os
import statistics
import subprocess
import time
import urllib.request

import pandas as pd
import streamlit as st

import access
import colab_notebook
import db
import evals_lib
import ftloop
import model_lab
import seed_data
import stats

LOCAL_MODELS = ["phi-4-14b", "qwen2.5-coder-7b-instruct", "mistralai/ministral-3-14b-reasoning",
                "llama-3.3-8b-instruct-i1", "gemma-4-e4b-it", "deepseek-r1-distill-qwen-14b"]
SINGLE_METRICS = ["Correctness (single judge)", "Faithfulness (single judge)", "Verifier (code tests, Docker)"]
LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "telar-logo.png")


@st.cache_data(ttl=3600, show_spinner=False)
def openrouter_free_models():
    """All $0 models on OpenRouter (needs OPENROUTER_API_KEY). Empty list if no key / unreachable."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return []
    try:
        req = urllib.request.Request("https://openrouter.ai/api/v1/models",
                                     headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r).get("data", [])
    except Exception:
        return []
    def free_text(m):
        p = m.get("pricing", {})
        if str(p.get("prompt", "1")) not in ("0", "0.0") or str(p.get("completion", "1")) not in ("0", "0.0"):
            return False  # not free
        outs = (m.get("architecture", {}) or {}).get("output_modalities") or ["text"]
        return set(outs) == {"text"}  # chat judge: text-only output (excludes audio/image/video models)
    return sorted(m["id"] for m in data if free_text(m))


@st.cache_data(ttl=60, show_spinner=False)
def lm_studio_models(api_base):
    """Live list of model ids loaded in LM Studio, so a freshly fine-tuned GGUF shows up
    (short TTL). Falls back to the hardcoded roster if LM Studio is unreachable."""
    try:
        with urllib.request.urlopen(api_base.rstrip("/") + "/models", timeout=5) as r:
            return sorted(m["id"] for m in json.load(r).get("data", []))
    except Exception:
        return []


def derive_base():
    if os.getenv("LM_STUDIO_API_BASE"):
        return os.getenv("LM_STUDIO_API_BASE")
    try:
        gw = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True, timeout=3).stdout.split()[2]
        return f"http://{gw}:1234/v1"
    except Exception:
        return "http://localhost:1234/v1"


def agree(res, gold):
    """Did a result match the gold label? Handles numeric (1-5) and text (FAITHFUL/...) gold."""
    g = (gold or "").strip()
    if not g:
        return None
    if g.replace(".", "", 1).isdigit() and isinstance(res.get("score"), (int, float)):
        return round(res["score"]) == round(float(g))
    if res.get("verdict"):
        return res["verdict"].upper() == g.upper()
    return None


_BADGE = {"PASS": ("#15331e", "#5ce08a"), "FAIL": ("#3a1620", "#ff6b8b"),
          "IMPROVE": ("#15331e", "#5ce08a"), "FLAT": ("#23262f", "#aab1c4"),
          "REGRESS": ("#3a1620", "#ff6b8b")}


def _badge(text):
    """A colored verdict pill (HTML; render with unsafe_allow_html=True)."""
    bg, fg = _BADGE.get(str(text).upper(), ("#23262f", "#cdd3e0"))
    return (f'<span style="background:{bg};color:{fg};border:1px solid {fg}55;border-radius:999px;'
            f'padding:.14rem .65rem;font-weight:700;font-size:.85rem;">{text}</span>')


def _style_verdicts(df):
    """Tint PASS/IMPROVE/FAITHFUL green and FAIL/REGRESS/UNFAITHFUL red in verdict-ish columns."""
    cols = [c for c in df.columns if c in ("verdict", "objective", "objective_verdict")]
    if not cols:
        return df
    def color(v):
        u = str(v).upper()
        if u in ("PASS", "IMPROVE", "FAITHFUL"):
            return "color:#5ce08a;font-weight:700"
        if u in ("FAIL", "REGRESS", "UNFAITHFUL", "ERROR"):
            return "color:#ff6b8b;font-weight:700"
        return ""
    sty = df.style
    fn = getattr(sty, "map", None)        # pandas >=2.1; fall back to the old name only if absent
    if fn is None:
        fn = sty.applymap
    return fn(color, subset=cols)


def _run_model_eval(test_ds, judge_metric, model_under_test, judge_model, jury_models, api_base, progress=None):
    """Model testing: the model-UNDER-TEST generates an answer for each test input, then a judge
    scores it. (Differs from the Run tab, which judges the dataset's stored output.) Stores a run."""
    cases = db.list_cases(test_ds)
    fn = evals_lib.METRICS[judge_metric]
    is_jury = judge_metric.startswith("Jury")
    is_verif = judge_metric.startswith("Verifier")
    rid = db.add_run(test_ds, f"{judge_metric} @ {model_under_test}", model_under_test, len(cases), "")
    n = len(cases) or 1
    # Phase 1 - the model under test is loaded once and generates ALL answers
    gens = []
    for i, c in enumerate(cases):
        try:
            gens.append(evals_lib.generate(model_under_test, api_base, c["input"], system=c.get("context") or ""))
        except Exception as e:  # noqa: BLE001
            gens.append(f"[generation error: {str(e)[:120]}]")
        if progress:
            progress.progress((i + 1) / (2 * n), text=f"generating {i+1}/{len(cases)}")
    # Phase 2 - the judge is loaded once and scores ALL of them (no per-case model swap)
    scores = []
    for i, (c, gen) in enumerate(zip(cases, gens)):
        try:
            if is_jury:
                res = fn(jury_models, api_base, question=c["input"], answer=gen)
            elif is_verif:
                res = fn(None, api_base, question=c["input"], answer=gen, context=c.get("context") or "")
            else:
                res = fn(judge_model, api_base, question=c["input"], answer=gen, context=c.get("context") or "")
        except Exception as e:  # noqa: BLE001
            res = {"score": None, "verdict": "ERROR", "reason": str(e)[:200], "tokens": 0, "secs": 0}
        db.add_result(rid, c["id"], res.get("score"), res.get("verdict", ""),
                      f"GENERATED: {gen[:400]}\n\nJUDGE: {res.get('reason', '')}"[:1200],
                      res.get("tokens", 0), res.get("secs", 0))
        if isinstance(res.get("score"), (int, float)):
            scores.append(res["score"])
        if progress:
            progress.progress((n + i + 1) / (2 * n), text=f"judging {i+1}/{len(cases)}")
    mean = sum(scores) / len(scores) if scores else None
    with db.conn() as cc:
        cc.execute("UPDATE runs SET summary=? WHERE id=?",
                   (f"mean {mean:.2f}" if mean is not None else "no numeric scores", rid))
    return rid, mean


def _ft_before_after(base_rid, after_rid):
    """Paired before/after verdict on the SAME test cases (IMPROVE only if CI lower bound clears 0)."""
    b = {r["case_id"]: r["score"] for r in db.get_results(base_rid) if isinstance(r["score"], (int, float))}
    a = {r["case_id"]: r["score"] for r in db.get_results(after_rid) if isinstance(r["score"], (int, float))}
    common = [k for k in b if k in a]
    if len(common) < 2:
        st.warning("Need at least 2 comparable scored cases in both runs to compare.")
        return
    bx, ax = [b[k] for k in common], [a[k] for k in common]
    cmp = stats.compare_two(ax, bx)        # candidate = after, baseline = before
    c1, c2, c3 = st.columns(3)
    c1.metric("Before (mean)", f"{sum(bx)/len(bx):.2f}")
    c2.metric("After (mean)", f"{sum(ax)/len(ax):.2f}", delta=f"{(sum(ax)/len(ax))-(sum(bx)/len(bx)):+.2f}")
    c3.metric("Cases compared", len(common))
    st.markdown(f"**Did fine-tuning help?** &nbsp; {_badge(cmp.verdict)}", unsafe_allow_html=True)
    st.markdown(f"Δ = {cmp.delta:+.3f} · 95% CI [{cmp.ci_lo:+.3f}, {cmp.ci_hi:+.3f}] · "
                f"p = {cmp.p:.4f} · {cmp.test} · n = {cmp.n}")
    st.caption("IMPROVE only if the CI lower bound clears zero — proof the gain is real, not judge noise.")


def _howto(md):
    """Per-tab plain-English explainer (collapsed by default)."""
    with st.expander("ℹ️ How this works", expanded=False):
        st.markdown(md)


def seed():
    rows = [
        ("examples", "Write a Python function is_even(n) returning True if n is even.",
         "def is_even(n):\n    return n % 2 == 0", "", "5"),
        ("examples", "Write a Python function is_even(n) returning True if n is even.",
         "def is_even(n):\n    return n % 2 == 1", "", "1"),
        ("examples", "Where is the Eiffel Tower and when was it completed?",
         "The Eiffel Tower is in Paris and was completed in 1889.",
         "The Eiffel Tower is in Paris, France, completed in 1889.", "FAITHFUL"),
        ("examples", "Where is the Eiffel Tower and when was it completed?",
         "The Eiffel Tower is in Berlin and was completed in 1950.",
         "The Eiffel Tower is in Paris, France, completed in 1889.", "UNFAITHFUL"),
    ]
    for r in rows:
        db.add_case(*r)


db.init()
st.set_page_config(page_title="Local LLM Eval Lab", page_icon="🧪", layout="wide")
if os.path.exists(LOGO):
    st.logo(LOGO, size="large")


def _require_password():
    """Gate the public dashboard: master admin (EVALS_DASHBOARD_PASSWORD, permanent) or a
    time-limited demo code (access.py). Fails closed if the admin password is unset."""
    admin_pw = os.getenv("EVALS_DASHBOARD_PASSWORD")
    if not admin_pw:
        st.error("Dashboard password not configured - set EVALS_DASHBOARD_PASSWORD.")
        st.stop()
    exp = st.session_state.get("_expires_at")          # expire a demo session in place
    if exp and time.time() > exp:
        for k in ("_authed", "_admin", "_expires_at"):
            st.session_state.pop(k, None)
        st.warning("Demo access expired - ask for a new password.")
    if st.session_state.get("_authed"):
        return
    _l, _c, _r = st.columns([1, 2, 1])
    with _c:
        st.markdown('<div style="height:6vh"></div>', unsafe_allow_html=True)
        if os.path.exists(LOGO):
            st.image(LOGO, width=340)
        st.markdown('<div class="hero-tag" style="font-size:1.55rem">Eval-driven AI engineering</div>'
                    '<div class="hero-sub">Free, local LLM-evaluation lab — 7 engines, $0, no cloud.</div>',
                    unsafe_allow_html=True)
        pw = st.text_input("Password", type="password", placeholder="demo password")
        if pw:
            if hmac.compare_digest(pw, admin_pw):
                st.session_state.update(_authed=True, _admin=True)
                st.rerun()
            elif (demo_exp := access.verify(pw)):
                st.session_state.update(_authed=True, _admin=False, _expires_at=demo_exp)
                st.rerun()
            else:
                st.error("Wrong or expired password.")
    st.stop()


def _admin_access_ui():
    """Sidebar panel (admin only): mint / list / revoke time-limited demo passwords."""
    label = st.text_input("Label (your note)", placeholder="recruiter-acme", key="da_label")
    ttl = st.selectbox("Valid for", ["1h", "8h", "1d", "3d", "7d"], index=2, key="da_ttl")
    if st.button("Create demo password", key="da_create"):
        pw, exp = access.add(label or "demo", ttl)
        st.success("Share this password (shown once):")
        st.code(pw)
        st.caption(f"Expires {datetime.datetime.fromtimestamp(exp):%Y-%m-%d %H:%M} · evals.telarlabs.online")
    codes = access.list_codes()
    if codes:
        st.caption("Active / recent:")
        for c in codes:
            col = st.columns([3, 2, 1])
            col[0].write(c["label"])
            col[1].caption(f"🟢 {c['hours_left']}h left" if c["active"] else "⚪ expired")
            if col[2].button("✕", key=f"rev_{c['id']}", help="revoke"):
                access.revoke(c["id"])
                st.rerun()


_require_password()

st.markdown(
    "<style>"
    ".stButton>button{border-radius:10px;padding:.5rem 1.1rem;font-weight:600;"
    "border:1px solid rgba(128,128,128,.35);transition:transform .12s ease,box-shadow .12s ease;}"
    ".stButton>button:hover{transform:translateY(-1px);box-shadow:0 3px 10px rgba(0,0,0,.18);border-color:#7c5cff;}"
    ".stButton>button[kind=\"primary\"]{background:linear-gradient(90deg,#7c5cff,#5b8cff);border:0;color:#fff;}"
    "div[data-baseweb=\"tab-list\"]{gap:4px;}"
    "button[data-baseweb=\"tab\"]{font-weight:600;border-radius:8px 8px 0 0;}"
    # --- hero / cards (Overview) ---
    ".hero-tag{font-size:2.15rem;font-weight:800;line-height:1.1;margin:.4rem 0 0;"
    "background:linear-gradient(90deg,#a98bff,#6ea0ff);-webkit-background-clip:text;"
    "-webkit-text-fill-color:transparent;}"
    ".hero-sub{color:#9aa0b4;font-size:1.02rem;margin:.35rem 0 1.1rem;max-width:46rem;}"
    ".cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:.2rem 0 1.3rem;}"
    ".card{background:#161b27;border:1px solid rgba(124,92,255,.22);border-radius:14px;padding:14px 16px;}"
    ".card .v{font-size:1.7rem;font-weight:800;color:#fff;line-height:1.1;}"
    ".card .l{color:#9aa0b4;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;margin-top:.2rem;}"
    ".egrid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;}"
    "@media(max-width:760px){.cards,.egrid{grid-template-columns:1fr;}}"
    ".ecard{background:rgba(22,27,39,.55);border:1px solid rgba(255,255,255,.07);border-radius:14px;"
    "padding:14px 16px;transition:border-color .15s ease,transform .15s ease;}"
    ".ecard:hover{border-color:rgba(124,92,255,.55);transform:translateY(-2px);}"
    ".ecard .t{font-weight:700;color:#eef0fb;font-size:1rem;}"
    ".ecard .d{color:#9aa0b4;font-size:.86rem;margin-top:.3rem;line-height:1.35;}"
    ".ecard .tag{display:inline-block;margin-top:.55rem;font-size:.72rem;color:#b3a4ff;"
    "border:1px solid rgba(124,92,255,.4);border-radius:999px;padding:.05rem .55rem;}"
    ".foot{color:#6b7180;font-size:.8rem;text-align:center;margin-top:1.6rem;}"
    # --- metric widgets -> cards (applies to every tab) ---
    "[data-testid=\"stMetric\"]{background:#161b27;border:1px solid rgba(124,92,255,.18);"
    "border-radius:14px;padding:12px 16px;}"
    "[data-testid=\"stMetricValue\"]{font-weight:800;}"
    "[data-testid=\"stMetricLabel\"]{opacity:.7;}"
    "button[data-baseweb=\"tab\"][aria-selected=\"true\"]{color:#b3a4ff;}"
    "h3{letter-spacing:-.01em;}"
    "</style>", unsafe_allow_html=True)

# Model registry: local (LM Studio, 🟢) + OpenRouter free models (🟣). Label -> model id passed to evals_lib.
OR_FREE = openrouter_free_models()
LOCAL_LABELS = {f"🟢 {m}": m for m in (lm_studio_models(derive_base()) or LOCAL_MODELS)}
OR_LABELS = {f"🟣 {m}": f"openrouter/{m}" for m in OR_FREE}
ALL_LABELS = {**LOCAL_LABELS, **OR_LABELS}
JURY_DEFAULT = [f"🟢 {m}" for m in evals_lib.DEFAULT_JURY if f"🟢 {m}" in LOCAL_LABELS]

with st.sidebar:
    st.header("⚙️ Config")
    with st.expander("🔌 Endpoint (advanced)"):
        api_base = st.text_input("LM Studio API base", derive_base())
    st.caption(f"🟢 {len(LOCAL_LABELS)} local · 🟣 {len(OR_LABELS)} OpenRouter free"
               + ("" if OR_FREE else " — set OPENROUTER_API_KEY to load free models"))
    single_label = st.selectbox("Single-judge model", list(ALL_LABELS), index=0)
    single_model = ALL_LABELS[single_label]
    jury_labels = st.multiselect("Jury panel", list(ALL_LABELS), default=JURY_DEFAULT)
    jury_models = [ALL_LABELS[l] for l in jury_labels]
    if st.button("Check LM Studio", width="stretch"):
        try:
            with urllib.request.urlopen(api_base.rstrip("/") + "/models", timeout=5) as r:
                n = len(json.load(r).get("data", []))
            st.success(f"OK - {n} models loaded")
        except Exception as e:
            st.error(f"Cannot reach LM Studio: {e}")
    st.divider()
    st.caption("🟢 local (LM Studio) · 🟣 OpenRouter free models — both $0. No cloud, no spend.")

if st.session_state.get("_admin"):
    with st.sidebar:
        st.divider()
        with st.expander("🔑 Demo access (admin)", expanded=False):
            _admin_access_ui()

st.title("🧪 Local LLM Eval Lab")
st.caption("Insert test cases · run evals against local models · see scores, reasoning, and cost.")

(tab_over, tab_ft, tab_data, tab_run, tab_live, tab_cmp, tab_traj, tab_lab, tab_res) = st.tabs(
    ["🏠 Overview", "🎓 Fine-tune", "📋 Data", "▶️ Run", "🔴 Live", "⚖️ Compare",
     "🧭 Trajectory", "🔬 Model lab", "📊 Results"])

# ---------------- Overview ----------------
with tab_over:
    cases, runs = db.list_cases(), db.list_runs()
    if os.path.exists(LOGO):
        st.image(LOGO, width=300)
    st.markdown('<div class="hero-tag">Eval-driven AI engineering</div>'
                '<div class="hero-sub">Seven free, local eval engines that measure agents honestly — '
                '$0, no cloud, no vendor lock-in.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="cards">'
        '<div class="card"><div class="v">7</div><div class="l">Eval engines</div></div>'
        f'<div class="card"><div class="v">{len(cases)}</div><div class="l">Test cases</div></div>'
        f'<div class="card"><div class="v">{len(runs)}</div><div class="l">Eval runs</div></div>'
        '<div class="card"><div class="v">$0</div><div class="l">Cost · 100% local</div></div>'
        '</div>', unsafe_allow_html=True)
    engines = [
        ("🧭 Trajectory", "Scores the agent's PATH — legal state transitions, tool args, goal completion. No LLM, $0.", "Trajectory"),
        ("⚖️ Compare + stats", "Judge vs judge with BCa bootstrap CI and paired tests. Ship on the lower bound, not the point estimate.", "Compare"),
        ("🛡️ Prompt-injection", "Honeypot trip-log → deterministic attack-success rate. 0.33 → 0.0 with a sandwich defense.", None),
        ("🧑 Simulated-user", "An LLM persona drives the agent; objective booking check + jury judged on-policy.", None),
        ("🔎 Judge-bias", "Position / verbosity / sycophancy probes. llama-3.3 +1.12 primacy; phi-4 and gemma clean.", None),
        ("🗂️ Failure taxonomy", "Auto-labels real agent failures into a closed set. Cohen's kappa 0.84 vs human.", None),
        ("🔁 Regression CI-gate", "Objective label-agreement vs a pinned baseline. Fails the build on a regression.", None),
        ("🔬 Model lab", "Whole-model red-team (garak) + GSM8K capability benchmark against LM Studio.", "Model lab"),
    ]
    grid = "".join(
        f'<div class="ecard"><div class="t">{name}</div><div class="d">{desc}</div>'
        + (f'<div class="tag">▸ {tab} tab</div>' if tab else "")
        + "</div>"
        for name, desc, tab in engines)
    st.markdown(f'<div class="egrid">{grid}</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)
    st.info("**Honest by design** — a +0.05 gap between two judges? The 95% CI calls it noise, not a win. "
            "The lab is built to *not* fool you.")
    if runs:
        st.markdown("**Recent runs**")
        st.dataframe(pd.DataFrame(runs)[["id", "eval_type", "dataset", "model", "n", "summary"]].head(8),
                     width="stretch", hide_index=True)
    st.markdown('<div class="foot">FreeStack Evals · built by Alex Tann · LM Studio + free OpenRouter · $0, 100% local</div>',
                unsafe_allow_html=True)

# ---------------- Fine-tune Loop ----------------
with tab_ft:
    st.subheader("🎓 Fine-tune Loop — prove a fine-tune helps, on YOUR data")
    with st.expander("ℹ️ What is this? (read me first)", expanded=False):
        st.markdown(
            "**The loop:** take a dataset → measure the base model (**before**) → fine-tune it on your "
            "data (free Google Colab, since the local GPU is AMD) → measure again (**after**) → let the "
            "stats say whether the gain is **real or just noise**. That proof is the whole point — anyone "
            "can fine-tune; few can prove it actually helped.\n\n"
            "**Why split train/test?** The model trains on `train` and is judged on `test`, which it never "
            "saw. Judging on data it trained on measures memorisation, not skill.\n\n"
            "**Why 'generate then judge'?** To test a *model* (not a fixed dataset answer), the model writes "
            "its own answer to each test input, then a judge scores that answer — same test set before & after.")
    st.markdown("**0 Split** → **1 Baseline (before)** → 2 Fine-tune on Colab → **3 Eval (after)** "
                "→ **4 Before/After verdict** → 5 Agent *(next)*")
    st.divider()

    base_dss = [d for d in db.datasets() if not ftloop.is_split(d)]
    if not base_dss:
        st.info("Add a dataset on the **Data** tab first (or click '📦 Load 50 real cases' there).")
    else:
        ds = st.selectbox("Dataset", base_dss, key="ft_ds")

        st.markdown("#### 0 · Split into train / test")
        st.caption("The model learns from `train`; we only ever judge on the held-out `test`.")
        ctr = len(db.list_cases(ds))
        sc = st.columns([2, 1])
        tf = sc[0].slider("Test fraction", 0.1, 0.5, 0.2, 0.05, key="ft_tf")
        if sc[1].button(f"Split {ctr} cases", key="ft_split"):
            tr, te, ntr, nte = ftloop.split(db, ds, tf)
            st.success(f"Split → {ntr} train · {nte} test")
            st.rerun()
        train_ds, test_ds = f"{ds}__train", f"{ds}__test"
        n_tr, n_te = len(db.list_cases(train_ds)), len(db.list_cases(test_ds))

        if not n_te:
            st.warning("👉 Click **Split** above first. Then the **judge metric (incl. Persona-fit)**, "
                       "the **Base model** picker, and **▶️ Run baseline** appear right here.")
        else:
            s1, s2 = st.columns(2)
            s1.metric("Train cases", n_tr)
            s2.metric("Test cases", n_te)
            st.download_button("⬇️ Export train as JSONL (feed this to the Colab fine-tune)",
                               ftloop.train_jsonl(db, train_ds), f"{ds}_train.jsonl",
                               "application/jsonl", key="ft_dl")

            judge_metric = st.selectbox("Judge metric (how each generated answer is scored)",
                                        list(evals_lib.METRICS), key="ft_metric")

            st.markdown("#### 1 · Baseline (before) — the BASE model answers the test set, a judge scores it")
            b1, b2 = st.columns([2, 1])
            base_label = b1.selectbox("Base model (under test)", list(ALL_LABELS), key="ft_base")
            if b2.button("▶️ Run baseline", key="ft_base_run_btn"):
                prog = st.progress(0.0, text="generating + judging...")
                rid, mean = _run_model_eval(test_ds, judge_metric, ALL_LABELS[base_label],
                                            single_model, jury_models, api_base, progress=prog)
                st.session_state["ft_base_run"] = rid
                st.success(f"Baseline run #{rid} — mean {mean:.2f}" if mean is not None else f"Baseline run #{rid}")

            st.markdown("#### 2 · Fine-tune on Colab — generate your notebook")
            st.caption("The local GPU is AMD (Unsloth needs NVIDIA), so training runs on a **free Google "
                       "Colab T4**. This makes a ready-to-run notebook: open it in Colab, upload the train "
                       "JSONL above, Run all, download the GGUF, import it into LM Studio.")
            f1, f2 = st.columns(2)
            nb_base = f1.text_input("Base model (Hugging Face id)", "unsloth/Qwen2.5-7B-Instruct",
                                    key="ft_nb_base", help="Match the family of the model you eval. "
                                    "e.g. unsloth/Llama-3.2-3B-Instruct, unsloth/Qwen2.5-Coder-7B-Instruct")
            nb_out = f2.text_input("Output model name", f"{ds}-ft", key="ft_nb_out")
            f3, f4 = st.columns(2)
            nb_epochs = f3.number_input("Epochs", 1, 5, 2, key="ft_nb_epochs")
            nb_quant = f4.selectbox("GGUF quantization", ["q4_k_m", "q5_k_m", "q8_0", "f16"], key="ft_nb_quant")
            st.download_button(
                "⬇️ Generate Colab notebook (.ipynb)",
                colab_notebook.build(base_model=nb_base, out_name=nb_out or "my-finetune",
                                     epochs=int(nb_epochs), quant=nb_quant,
                                     train_filename=f"{ds}_train.jsonl"),
                f"finetune_{ds}.ipynb", "application/json", key="ft_nb_dl", type="primary")
            st.caption("Open at colab.research.google.com (File → Upload notebook) → Runtime → T4 GPU → Run all. "
                       "Every step in the notebook is explained for first-timers.")

            st.markdown("#### 3 · Eval (after) — the FINE-TUNED model answers the SAME test set")
            a1, a2 = st.columns([2, 1])
            after_label = a1.selectbox("Fine-tuned model (load its GGUF into LM Studio first)",
                                       list(ALL_LABELS), key="ft_after")
            if a2.button("▶️ Run after", key="ft_after_run_btn"):
                prog = st.progress(0.0, text="generating + judging...")
                rid, mean = _run_model_eval(test_ds, judge_metric, ALL_LABELS[after_label],
                                            single_model, jury_models, api_base, progress=prog)
                st.session_state["ft_after_run"] = rid
                st.success(f"After run #{rid} — mean {mean:.2f}" if mean is not None else f"After run #{rid}")

            st.markdown("#### 4 · Before / After — did it really help?")
            br, ar = st.session_state.get("ft_base_run"), st.session_state.get("ft_after_run")
            if br and ar:
                _ft_before_after(br, ar)
            else:
                st.caption("Run both the baseline and the after eval to see the paired verdict.")

# ---------------- Data ----------------
with tab_data:
    _howto("Your **test cases** live here — each is a question (*input*), a candidate answer (*output*), "
           "optional *context*, and an optional *gold label* (the known-correct answer). Add them by hand, "
           "import a CSV, or click **Load 50 real cases** for a ready mixed set. Everything else in the lab "
           "runs on these cases.")
    c1, c2 = st.columns([2, 1])
    with c1:
        with st.form("addcase", clear_on_submit=True):
            ds = st.text_input("Dataset", "demo")
            inp = st.text_area("Input / question")
            out = st.text_area("Output / candidate answer (for Verifier: the code)")
            ctx = st.text_area("Context (faithfulness source, or Verifier pytest tests)")
            gold = st.text_input("Gold label (optional: 1-5, or FAITHFUL/UNFAITHFUL)")
            if st.form_submit_button("Add case"):
                if inp and out:
                    db.add_case(ds, inp, out, ctx, gold)
                    st.success("Case added")
                else:
                    st.warning("Input and output are required")
    with c2:
        if st.button("📦 Load 50 real cases", type="primary", width="stretch"):
            if "real-faithfulness" in db.datasets():
                st.info("50 real cases already loaded.")
            else:
                n = seed_data.load(db)
                st.success(f"Loaded {n} real cases (faithfulness / correctness / verifier)")
                st.rerun()
        if st.button("Load example dataset", width="stretch"):
            seed()
            st.success("Seeded 'examples'")
            st.rerun()
        up = st.file_uploader("Import CSV", type="csv",
                              help="columns: dataset,input,output,context,gold")
        if up is not None:
            idf = pd.read_csv(up).fillna("")
            for _, row in idf.iterrows():
                db.add_case(str(row.get("dataset", "imported")), str(row.get("input", "")),
                            str(row.get("output", "")), str(row.get("context", "")), str(row.get("gold", "")))
            st.success(f"Imported {len(idf)} cases")
            st.rerun()
    st.divider()
    cases = db.list_cases()
    if cases:
        st.dataframe(pd.DataFrame(cases)[["id", "dataset", "input", "output", "context", "gold"]],
                     width="stretch", height=280, hide_index=True)
        st.markdown("**Delete cases**")
        d1, d2, d3 = st.columns([3, 1, 2])
        case_labels = {f"#{c['id']} [{c['dataset']}] {c['input'][:60]}": c["id"] for c in cases}
        chosen_label = d1.selectbox("Select case to delete", list(case_labels))
        if d2.button("Delete selected"):
            db.delete_case(case_labels[chosen_label])
            st.rerun()
        all_ds = sorted({c["dataset"] for c in cases})
        del_ds = d3.selectbox("Or delete entire dataset", ["— pick —"] + all_ds, key="del_ds")
        if del_ds != "— pick —" and st.button(f"Delete all '{del_ds}' cases", type="secondary"):
            for c in [x for x in cases if x["dataset"] == del_ds]:
                db.delete_case(c["id"])
            st.rerun()
    else:
        st.info("No cases yet - add one or click 'Load example dataset'.")

# ---------------- Run ----------------
with tab_run:
    _howto("Score every case in a dataset with a judge. A **single judge** is one model grading 1–5; a "
           "**jury** (pick the panel in the sidebar) is several models voting by median, so one biased judge "
           "can't decide alone. **Verifier** skips the model and runs the candidate code against real tests "
           "in a sandbox — objective ground truth, no opinion.")
    dss = db.datasets()
    if not dss:
        st.info("Add data on the Data tab first.")
    else:
        ds = st.selectbox("Dataset", dss, key="run_ds")
        metric_name = st.selectbox("Eval metric", list(evals_lib.METRICS), key="run_metric")
        cases = db.list_cases(ds)
        is_jury = metric_name.startswith("Jury")
        is_verif = metric_name.startswith("Verifier")
        if is_verif:
            st.info("Verifier: **Output** = candidate code, **Context** = pytest tests that `from solution import ...`. Objective, no model.")
        st.write(f"**{len(cases)}** cases · "
                 f"{'jury ['+', '.join(m.split('/')[-1] for m in jury_models)+']' if is_jury else ('Docker sandbox' if is_verif else single_model)}")
        if is_jury and not jury_models:
            st.warning("Pick at least one jury model in the sidebar.")
        elif st.button("▶️ Run eval", type="primary"):
            fn = evals_lib.METRICS[metric_name]
            run_model = ",".join(m.split("/")[-1] for m in jury_models) if is_jury else ("docker" if is_verif else single_model)
            rid = db.add_run(ds, metric_name, run_model, len(cases), "")
            prog = st.progress(0.0, text="running...")
            rows = []
            for i, c in enumerate(cases):
                try:
                    if is_jury:
                        res = fn(jury_models, api_base, question=c["input"], answer=c["output"])
                    else:
                        res = fn(single_model, api_base, question=c["input"], answer=c["output"],
                                 context=c.get("context") or "")
                except Exception as e:  # noqa: BLE001
                    res = {"score": None, "verdict": "ERROR", "reason": str(e)[:200], "tokens": 0, "secs": 0}
                db.add_result(rid, c["id"], res.get("score"), res.get("verdict", ""),
                              res.get("reason", ""), res.get("tokens", 0), res.get("secs", 0))
                rows.append({"input": c["input"][:45], **{k: res.get(k) for k in ("score", "verdict", "tokens", "secs")}})
                prog.progress((i + 1) / len(cases), text=f"{i+1}/{len(cases)}")
            scores = [r["score"] for r in rows if isinstance(r["score"], (int, float))]
            summ = f"mean {sum(scores)/len(scores):.2f}" if scores else "no numeric scores"
            with db.conn() as cc:
                cc.execute("UPDATE runs SET summary=? WHERE id=?", (summ, rid))
            st.success(f"Run #{rid} complete - {summ}. Open the Results tab for model responses.")
            st.dataframe(_style_verdicts(pd.DataFrame(rows)), width="stretch", hide_index=True)

# ---------------- Live (watch each judge think) ----------------
with tab_live:
    _howto("Same scoring as **Run**, but one case at a time: each judge **streams its reasoning** as it "
           "thinks, then reveals a score. Use it to see *why* a model grades the way it does — and to spot a "
           "judge that talks itself into a wrong score.")
    st.subheader("🔴 Live run - watch each judge think, step by step")
    st.caption("Pick one case and a panel; each judge streams its reasoning live, then reveals a score - "
               "the model's process under the hood. Mix 🟢 local and 🟣 OpenRouter judges.")
    dss = db.datasets()
    if not dss:
        st.info("Add data first (Data tab) - or click '📦 Load 50 real cases'.")
    else:
        lds = st.selectbox("Dataset", dss, key="live_ds")
        lcases = db.list_cases(lds)
        if not lcases:
            st.info("No cases in this dataset.")
        else:
            opt = {f"#{c['id']} · {c['input'][:70]}": c for c in lcases}
            chosen = opt[st.selectbox("Case", list(opt), key="live_case")]
            live_panel = st.multiselect("Judges", list(ALL_LABELS),
                                        default=(jury_labels[:3] or [single_label]), key="live_panel")
            with st.container(border=True):
                st.markdown(f"**Question:** {chosen['input']}")
                st.markdown(f"**Candidate answer:** {chosen['output']}")
                if chosen.get("context"):
                    st.caption(f"Context: {chosen['context'][:240]}")
            if st.button("🔴 Run live", type="primary", width="stretch") and live_panel:
                scores = []
                for lbl in live_panel:
                    with st.container(border=True):
                        st.markdown(f"**{lbl}**  ·  _thinking…_")
                        try:
                            full = st.write_stream(evals_lib.stream_judge(
                                ALL_LABELS[lbl], api_base, chosen["input"], chosen["output"],
                                chosen.get("context") or ""))
                            sc = evals_lib.parse_score(full)
                            scores.append(sc)
                            st.success(f"→ score = {sc}" if sc is not None else "→ (no score parsed)")
                        except Exception as e:  # noqa: BLE001
                            st.error(str(e)[:140])
                nums = [s for s in scores if isinstance(s, int)]
                if nums:
                    spread = max(nums) - min(nums)
                    st.divider()
                    c1, c2 = st.columns(2)
                    c1.metric("Panel median", statistics.median(nums))
                    c2.metric("Spread", spread,
                              "judges disagree" if spread >= 2 else "judges agree", delta_color="inverse")

# ---------------- Compare ----------------
with tab_cmp:
    _howto("Put models head-to-head on the **same** cases. The lab doesn't just show two averages — it asks "
           "**is the gap real or noise?** with a paired statistical test and a confidence interval. Verdict "
           "**IMPROVE / FLAT / REGRESS**; FLAT means the difference could be luck on a small sample. This is "
           "the honest way to claim one model (or a fine-tune) is better.")
    st.subheader("Compare judge models on a dataset")
    st.caption("Same task, different judges - watch quality vs token cost. (The single-vs-jury story, interactive.)")
    dss = db.datasets()
    if not dss:
        st.info("Add data first.")
    else:
        ds = st.selectbox("Dataset", dss, key="cmp_ds")
        metric_name = st.selectbox("Metric", ["Correctness (single judge)", "Faithfulness (single judge)"], key="cmp_metric")
        cmp_labels = st.multiselect("Models to compare", list(ALL_LABELS), default=[single_label], key="cmp_models")
        cmp_models = [ALL_LABELS[lbl] for lbl in cmp_labels]
        cases = db.list_cases(ds)
        st.write(f"{len(cases)} cases × {len(cmp_models)} models")
        if st.button("⚖️ Run comparison", type="primary") and cmp_models:
            fn = evals_lib.METRICS[metric_name]
            binary_metric = metric_name.startswith("Faithfulness")
            prog = st.progress(0.0)
            total = max(1, len(cmp_models) * len(cases))
            done, rows, per_model = 0, [], {}
            for m in cmp_models:
                short = m.split("/")[-1]
                aligned, tok, sec, ag, ng = [], 0, 0.0, 0, 0
                for c in cases:
                    try:
                        r = fn(m, api_base, question=c["input"], answer=c["output"], context=c.get("context") or "")
                    except Exception:  # noqa: BLE001
                        r = {"score": None, "tokens": 0, "secs": 0}
                    s = r.get("score")
                    aligned.append(s if isinstance(s, (int, float)) else None)
                    tok += r.get("tokens", 0) or 0
                    sec += r.get("secs", 0) or 0
                    a = agree(r, c.get("gold"))
                    if a is not None:
                        ng += 1
                        ag += int(a)
                    done += 1
                    prog.progress(done / total)
                per_model[short] = aligned
                vals = [s for s in aligned if s is not None]
                ci = stats.mean_ci(vals) if vals else None
                rows.append({"model": short, "_mean": ci.point if ci else None,
                             "mean [95% CI]": ci.fmt() if ci else "-",
                             "agreement w/ gold": f"{ag}/{ng}" if ng else "-",
                             "tokens": tok, "secs": round(sec, 1)})
            cdf = pd.DataFrame(rows)
            st.dataframe(cdf.drop(columns=["_mean"]), width="stretch", hide_index=True)
            cc1, cc2 = st.columns(2)
            if cdf["_mean"].notna().any():
                cc1.markdown("**Mean score**")
                cc1.bar_chart(cdf.set_index("model")["_mean"], color="#7c5cff")
            cc2.markdown("**Token cost**")
            cc2.bar_chart(cdf.set_index("model")["tokens"], color="#5b8cff")

            # paired significance: is the gap between the first two judges real, or noise?
            if len(cmp_models) >= 2:
                s0, s1 = rows[0]["model"], rows[1]["model"]
                paired = [(x, y) for x, y in zip(per_model[s0], per_model[s1])
                          if x is not None and y is not None]
                if len(paired) >= 2:
                    cmp = stats.compare_two([p[0] for p in paired], [p[1] for p in paired],
                                            binary=binary_metric)
                    st.markdown(f"**Is the gap real? {s0} vs {s1}** &nbsp; {_badge(cmp.verdict)}",
                                unsafe_allow_html=True)
                    st.markdown(f"Δ = {cmp.delta:+.3f}  ·  95% CI [{cmp.ci_lo:+.3f}, {cmp.ci_hi:+.3f}]  "
                                f"·  p = {cmp.p:.4f}  ·  {cmp.test}  ·  n = {cmp.n}")
                    st.caption("Ship on the CI lower bound, not the point estimate. "
                               "FLAT = the gap could be judge noise (the honest small-sample finding).")

# ---------------- Trajectory ----------------
with tab_traj:
    _howto("For **agents**, the final answer isn't the only thing — the **path** matters. This scores whether "
           "the agent took legal steps, used the right tool arguments, and reached the goal — with **no LLM**, "
           "just deterministic rules ($0). The soft layer adds a jury on the *words*; the gap between them "
           "catches a polite transcript that still hides an illegal action.")
    st.subheader("🧭 Trajectory eval - score the agent's PATH, not just its words")
    st.caption("Objective layer (no LLM, $0): legal state transitions + tool args + goal completion of a "
               "recorded PulseSales run. Soft layer: the free 5-family jury rates the words. The gap between "
               "them is the point - a polite transcript can still hide an illegal action.")
    import sys as _sys
    from pathlib import Path as _Path
    _td = _Path(__file__).resolve().parents[1] / "evals" / "trajectory-eval"
    if str(_td) not in _sys.path:
        _sys.path.insert(0, str(_td))
    try:
        import trace_schema as _ts
        import check_trajectory as _ct
        traces = _ts.load_traces(_td / "gold_traces.json")
        per = [_ct.check_trace(t) for t in traces]
        n = len(per)
        n_pass = sum(1 for r in per if r["verdict"] == "PASS")
        illegal = [r for r in per if r["n_illegal"] > 0]
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Traces", n)
        o2.metric("PASS / FAIL", f"{n_pass} / {n - n_pass}")
        o3.metric("Illegal traces", len(illegal))
        o4.metric("Mean legal-transition",
                  f"{stats.mean_ci([r['legal_transition_rate'] for r in per]).point:.2f}")

        st.markdown("**Objective (action path) - no LLM, deterministic ground truth**")
        objdf = pd.DataFrame([{
            "id": r["id"], "label": r["label"], "verdict": r["verdict"],
            "legal_trans": r["legal_transition_rate"], "tool_args": r["tool_arg_valid_rate"],
            "goal": r["goal_completion"], "efficiency": r["step_efficiency"], "n_illegal": r["n_illegal"],
            "first_illegal": (r["illegal"][0]["reasons"][0] if r["illegal"] else "")} for r in per])
        st.dataframe(_style_verdicts(objdf), width="stretch", hide_index=True)

        st.markdown("**Soft (words) jury + code-vs-jury agreement**")
        runs = sorted((_td / "runs").glob("soft_*.json"))
        if runs:
            soft = json.loads(runs[-1].read_text())
            s = soft["summary"]
            st.write(f"Latest jury run `{runs[-1].name}` - code-vs-jury agreement **{s['code_vs_jury_agreement']}**"
                     f", disagreements: **{', '.join(s['disagreements']) or 'none'}**")
            sdf = pd.DataFrame([{
                "id": p["id"], "label": p["label"], "objective": p["objective_verdict"],
                "jury_median (words)": p["soft_median"], "agree": p["agree"]} for p in soft["per_trace"]])
            st.dataframe(_style_verdicts(sdf), width="stretch", hide_index=True)
            for p in [p for p in soft["per_trace"] if not p["agree"]]:
                st.warning(f"⚠️ **{p['id']}**: jury rated the words ~{p['soft_median']} "
                           f"(looks {'fine' if p['soft_pass'] else 'bad'}) but the objective layer says "
                           f"**{p['objective_verdict']}** ({p['n_illegal']} illegal step) - the action-layer bug "
                           "a text-only judge cannot see.")
        else:
            st.info("No soft jury run yet - click below or run "
                    "`source env-wsl.sh && dashboard/.venv/bin/python evals/trajectory-eval/score_soft.py`.")
        if st.button("⚖️ Score soft steps with jury (LM Studio, ~2-4 min)"):
            import subprocess as _sp
            with st.spinner("Running the 3-family jury over transcripts (judge-outer, sequential)..."):
                pr = _sp.run([_sys.executable, str(_td / "score_soft.py")],
                             capture_output=True, text=True, cwd=str(_td.parents[1]))
            st.code((pr.stdout or "")[-2000:] or (pr.stderr or "")[-2000:])
            st.rerun()
    except Exception as e:  # noqa: BLE001
        st.error(f"Trajectory eval error: {e}")

# ---------------- Model lab ----------------
with tab_lab:
    _howto("Whole-model checks, not per-case: a **security red-team** (garak throws known attacks and sees if "
           "the model complies) and a **capability benchmark** (GSM8K grade-school math). Runs against your "
           "local LM Studio model. Slower, so it's on-demand.")
    st.subheader("🔬 Model lab - whole-model evals")
    st.caption("Scan/benchmark an entire model (not per-case). Slower; runs on demand.")
    lab_label = st.selectbox("Model", list(LOCAL_LABELS), key="lab_model")
    lab_model = LOCAL_LABELS[lab_label]
    st.caption("Model lab uses local models only (garak / lm-eval run against LM Studio).")
    a, b = st.columns(2)
    with a:
        st.markdown("**🛡️ Security red-team (garak)**")
        probe = st.selectbox("Probe", ["dan.Dan_11_0", "dan.AntiDAN", "glitch.Glitch"], index=0)
        if st.button("Run security scan"):
            with st.spinner("Running garak (~15-60s)..."):
                r = model_lab.garak_scan(lab_model, api_base, probe)
            (st.code(r["text"]) if r["ok"] else st.warning(r["text"]))
            if r.get("ok"):
                st.caption("FAIL = a vulnerability was found (the model complied with the attack).")
    with b:
        st.markdown("**📚 Capability benchmark (GSM8K)**")
        lim = st.slider("Questions", 5, 50, 10)
        if st.button("Run GSM8K"):
            with st.spinner(f"Running GSM8K × {lim} (can take minutes)..."):
                r = model_lab.gsm8k_bench(lab_model, api_base, lim)
            (st.code(r["text"]) if r["ok"] else st.warning(r["text"]))

# ---------------- Results ----------------
with tab_res:
    _howto("Every past run: the scores, the **mean with its confidence interval** (a wide interval = a shaky "
           "number from too few cases), agreement with gold labels, and — in the expanders — **exactly what "
           "each model said**. Download any run as CSV.")
    runs = db.list_runs()
    if not runs:
        st.info("No runs yet - run an eval first.")
    else:
        labels = {f"#{r['id']} · {r['eval_type']} · {r['dataset']} · {r['summary']}": r["id"] for r in runs}
        rid = labels[st.selectbox("Run", list(labels))]
        res = db.get_results(rid)
        df = pd.DataFrame(res)
        scores = [r["score"] for r in res if isinstance(r["score"], (int, float))]
        mci = stats.mean_ci(scores) if scores else None
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cases", len(res))
        m2.metric("Mean score", f"{mci.point:.2f}" if mci else "-",
                  help=(f"95% CI [{mci.lo:.2f}, {mci.hi:.2f}] · n={mci.n} ({mci.method})" if mci else None))
        m3.metric("Total tokens", int(df["tokens"].sum()) if len(df) else 0)
        m4.metric("Total time (s)", f"{df['secs'].sum():.0f}" if len(df) else 0)
        if mci:
            st.caption(f"Mean 95% CI [{mci.lo:.2f}, {mci.hi:.2f}] · n={mci.n} — a wide interval on few cases "
                       "means the number is noisy (judge variance), not a solid estimate.")
        ag = [agree(r, r.get("gold")) for r in res]
        ag = [x for x in ag if x is not None]
        if ag:
            st.metric("Agreement with gold", f"{sum(ag)}/{len(ag)} ({100*sum(ag)/len(ag):.0f}%)")
        if scores:
            st.bar_chart(df.set_index("case_id")["score"], color="#7c5cff")
        st.download_button("⬇️ Download results CSV", df.to_csv(index=False).encode(), f"run_{rid}.csv", "text/csv")
        st.markdown("#### 🔎 Model responses (what each model actually said)")
        for r in res:
            head = f"Case {r['case_id']}  ·  score = {r['score']}" + (f"  ·  {r['verdict']}" if r.get("verdict") else "")
            with st.expander(head):
                st.markdown(f"**Input:** {r.get('input', '')}")
                st.markdown(f"**Candidate output:** {r.get('output', '')}")
                if r.get("gold"):
                    st.markdown(f"**Gold label:** {r['gold']}")
                st.markdown("**Model response / reasoning:**")
                st.markdown(r.get("reason") or "_(no reasoning returned)_")
