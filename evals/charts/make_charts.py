"""Generate portfolio charts from the FreeStack eval results (all numbers measured today).
Saves PNGs to evals/charts/. Run with a venv that has matplotlib:
    uv pip install matplotlib && evals/charts/.venv/bin/python evals/charts/make_charts.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
BLUE, RED, GREY = "#2b6cb0", "#c53030", "#a0aec0"

# ---- Chart 1: the honest headline - jury vs single, equal accuracy, ~4.7x cost ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))
ax1.bar(["single\n(phi-4)", "5-family\njury"], [100, 100], color=[GREY, BLUE])
ax1.set_ylim(0, 110); ax1.set_ylabel("agreement with gold (%)")
ax1.set_title("Accuracy (held-out test, 18 items)")
for i, v in enumerate([100, 100]):
    ax1.text(i, v + 2, f"{v}%", ha="center", fontweight="bold")
ax2.bar(["single\n(phi-4)", "5-family\njury"], [4549, 21302], color=[GREY, RED])
ax2.set_ylabel("tokens used"); ax2.set_title("Cost (tokens)")
for i, v in enumerate([4549, 21302]):
    ax2.text(i, v + 400, f"{v:,}", ha="center", fontweight="bold")
ax2.text(1, 21302 / 2, "~4.7x", ha="center", color="white", fontweight="bold", fontsize=13)
fig.suptitle("5-model jury vs single judge: same accuracy, ~4.7x the cost\n"
             "(summary faithfulness - honest result: one good judge suffices here)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT / "1_jury_cost_vs_accuracy.png", dpi=130)

# ---- Chart 2: the toolkit discriminates good code/answers from bad ----
disc = {  # engine: (good %, bad %)
    "Verifier\n(pytest)": (100, 33),
    "DeepEval\n(G-Eval)": (100, 2),
    "Jury\n(1-5)": (100, 20),
    "RAGAS\n(faithfulness)": (100, 50),
}
fig, ax = plt.subplots(figsize=(8, 4.4))
x = range(len(disc))
good = [v[0] for v in disc.values()]
bad = [v[1] for v in disc.values()]
ax.bar([i - 0.2 for i in x], good, 0.4, label="correct / faithful input", color=BLUE)
ax.bar([i + 0.2 for i in x], bad, 0.4, label="buggy / hallucinated input", color=RED)
ax.set_xticks(list(x)); ax.set_xticklabels(disc.keys())
ax.set_ylabel("score (normalized %)"); ax.set_ylim(0, 110)
ax.set_title("Each engine separates good from bad (4 engines, local models, $0)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "2_eval_discrimination.png", dpi=130)

# ---- Chart 3: per-model judge agreement (the panel) ----
judges = {"phi-4\n(MS)": 18, "llama-3.3\n(Meta)": 18, "gemma-4\n(Google)": 18,
          "ministral\n(Mistral)": 17, "deepseek-r1\n(DeepSeek)": 17}
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(list(judges.keys()), list(judges.values()), color=BLUE)
ax.set_ylim(0, 19); ax.set_ylabel("correct / 18 (test split)")
ax.set_title("Per-model judge agreement - five decorrelated families")
for i, v in enumerate(judges.values()):
    ax.text(i, v + 0.2, str(v), ha="center", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "3_judge_agreement.png", dpi=130)

print("wrote:", *[p.name for p in sorted(OUT.glob("*.png"))])
