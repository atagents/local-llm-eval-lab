# FreeStack Eval Lab — a plain-English guide

For anyone, including people who have never evaluated or fine-tuned a model. No jargon left
undefined. The lab runs **100% free and local** (your own models in LM Studio + free cloud
for training). Live demo: **evals.telarlabs.online**.

## 1. The 30-second idea
An **AI model** can give wrong, made-up, or unsafe answers. Before you trust one in a product,
you **test** it — the same way you test code. This lab is the testing bench.

- **Eval** = a test for an AI ("how good are its answers on these examples?").
- **Judge** = an AI that grades another AI's answer (1–5, or pass/fail).
- **Jury** = several different judge models voting, so one biased judge can't decide alone.
- **Baseline** = the score *before* you change anything (your reference point).

You can't improve what you don't measure. And a small score change is often just **noise** —
the lab uses statistics to tell a **real** improvement from a lucky one.

## 2. Why it's different (the honest part)
Most demos show a number going up and say "better!". This lab proves whether the number is
**real**: it uses a **jury of different model families**, **statistics** (a confidence interval —
the gain only counts if even the pessimistic estimate is positive), checks the **agent's actions**
(not just its words), and tests **safety**. It is built to *not fool you*.

## 3. The dashboard, tab by tab
| Tab | In plain words |
|---|---|
| 🏠 Overview | The landing — what the lab is, the engines, headline results. |
| 🎓 Fine-tune | The full loop: split your data → measure the base model → (train on Colab) → measure again → prove it helped. |
| 📋 Data | Your test cases (question → expected answer). Add by hand, import CSV, or load 50 ready ones. |
| ▶️ Run | Score a dataset with a judge or jury. |
| 🔴 Live | Watch each judge "think" out loud, then reveal its score. |
| ⚖️ Compare | Put two models head-to-head and ask: is the gap **real or noise**? |
| 🧭 Trajectory | Score an **agent's path** (did it use the right tools, in a legal order, reach the goal?) — not just its words. |
| 🔬 Model lab | Whole-model checks: safety red-team (garak) + a capability benchmark (GSM8K math). |
| 📊 Results | Past runs, scores, and exactly what each model said. |

## 4. The Fine-tune Loop, step by step (the headline workflow)
Goal: take **your dataset**, make a model better at it, and **prove** the gain — then build an
agent on that model.

0. **Split** your data into `train` (the model learns from) and `test` (held-out — it never sees
   it). Judging on data it trained on would measure memorising, not skill.
1. **Baseline (before):** the base model answers the `test` questions; a judge scores them → your
   "before" number.
2. **Fine-tune:** train the model on the `train` set. *(LoRA = a cheap way to teach a model new
   behaviour by training a small add-on, not the whole thing.)* Your local GPU is **AMD**, and the
   common tool **Unsloth** needs an **NVIDIA** GPU — so you train on a **free Google Colab** notebook
   (still $0). Out comes a model file (**GGUF**) you load into LM Studio.
3. **After:** the fine-tuned model answers the **same** `test` questions → your "after" number.
4. **Proof:** the lab compares before vs after on the same cases. Verdict **IMPROVE / FLAT /
   REGRESS** — and it only says IMPROVE if the gain survives the statistics (not luck).
5. **Agent (next):** wrap the fine-tuned model in an agent (tools + steps) and test its **behaviour**
   (right tool path, safe, gets the user to the goal). Ship it as your chatbot.

Full process notes: [FINETUNE-LOOP.md](FINETUNE-LOOP.md).

## 5. What you need (the $0 stack)
- **LM Studio** (on your PC) — runs the models locally. The engine. **No key.**
- **Google Colab** — free cloud GPU for the fine-tune step. **No key** (just a Google login).
- **Hugging Face token** (`HF_TOKEN`) — only for fine-tuning: lets Colab download base models
  (some need sign-in) and optionally upload your trained model. Free.
- *Optional* **OpenRouter** / **DeepSeek** keys — add more free/cheap judge families for a more
  diverse jury. Nice to have, not required.

Nothing here costs money to run. See the model + key details in the README.

## 6. Mini-glossary
- **Model** — the AI that answers. **Base model** — before fine-tuning. **Fine-tuned** — after.
- **Token** — a chunk of text the model reads/writes; "token cost" ≈ how much it had to process.
- **Judge / Jury** — AI grader(s). **Gold label** — the known-correct answer for a case.
- **Baseline** — the before-score. **Confidence interval (CI)** — a range the true score likely sits
  in; a wide range = the number is shaky (few examples).
- **Fine-tune / LoRA** — teach a model your task; LoRA does it cheaply with a small add-on.
- **GGUF** — a model file format LM Studio loads.
- **Trajectory** — the sequence of actions an agent takes. **ASR** — attack success rate (lower = safer).
- **Agent** — a model that can take steps and use tools, not just answer once.
