# EVALS-CATALOG - FreeStack Harness Lab

*Каталог open-source эвал-фреймворков под бесплатный локальный стек (LM Studio + Docker), прицел на бизнес/продукты. Собран многоагентным обзором 2026-06-23 (6 категорий + проверка лицензий). Дополняет [harness-репо](GitHub-Repos-Harness-Engineering-2026-06.md) и [бизнес-репо](GitHub-Repos-Business-Solutions-2026-06.md) - тут именно про эвалы. Тело каталога на английском (имена и описания инструментов).*

## Измерено вживую: жюри vs одиночный судья (2026-06-23, честный результат)

Два прогона на ручных наборах summary-faithfulness (binary), наборы НЕ подгонялись под результат.
- **v2 (строгий, held-out split):** 30 кейсов, 12 калибровка / 18 тест; взвешенное голосование с весами ТОЛЬКО с калибровочного сплита (не циркулярно). На тест-сплите: одиночный phi-4 **18/18 (100%)**, жюри majority **18/18**, жюри weighted **18/18** → **жюри +0**. Цена жюри ~4.7x токенов. [jury/results-v2-2026-06-23.md](jury/results-v2-2026-06-23.md).
- **v1 (15 кейсов):** одиночный phi-4 14/15 (93%), жюри 15/15 (100%), +1 — но лучший одиночный тоже 100%. [jury/results-2026-06-22.md](jury/results-2026-06-22.md).
- **Честный вывод:** на бинарной faithfulness **один сильный локальный судья ≈ жюри из 5, при ~1/5 цены**. Жюри НЕ окупается, пока ошибки одиночного судьи редки. Премия жюри проявится только там, где даже лучший одиночный заметно ошибается: субъективные рубрики, состязательные кейсы, слабые модели. Это cost-aware вывод, а не «жюри всегда лучше».

## Рекомендованный стартовый набор (free + local)

1. **promptfoo** - Product CI / output-quality / red-team
   - `npm install -g promptfoo   (or: npx promptfoo@latest init)`
   - Lowest-friction, config-not-code (YAML) eval and regression runner with the best batteries-included local-model support (LM Studio via openai:chat:<model> + OPENAI_BASE_URL).

2. **DeepEval** - LLM-as-judge / output-quality / RAG / CI
   - `pip install -U deepeval   (set DEEPEVAL_TELEMETRY_OPT_OUT=YES)`
   - Largest reusable metric library (G-Eval custom rubrics, faithfulness, relevancy, hallucination, plus RAG and agent metrics) packaged as pytest assertions, so evals become CI gates with no new platform.

3. **RAGAS** - RAG evaluation
   - `pip install ragas`
   - De-facto standard RAG metric suite (faithfulness, answer/context relevancy, context precision/recall) plus synthetic test-set generation to bootstrap golden sets from your own docs.

4. **lm-evaluation-harness (EleutherAI)** - Capability benchmarks
   - `pip install lm-eval`
   - The de-facto standard runner for academic benchmarks (MMLU, GSM8K, IFEval, BBH) producing the canonical numbers stakeholders recognize.

5. **garak (NVIDIA)** - Safety / red-team / robustness
   - `pip install garak`
   - The de-facto LLM vulnerability scanner: 50+ pre-canned probes (jailbreaks, prompt injection, PII leakage, toxicity) with a batch report - a pure-CLI pre-ship security gate.

6. **Langfuse** - Observability / regression / tracing
   - `docker compose up   (clone langfuse/langfuse; pip install langfuse)`
   - Self-hostable tracing + datasets + LLM-as-judge + experiment comparison - adds persistence, a UI, and a production quality loop the test-runners lack.

---

## FreeStack Eval Catalog - Reusable AI Evals on a Free Local Stack

This catalog is for assembling many reusable **evals** that run **free and local** (LM Studio's OpenAI-compatible endpoint on the Windows host, plus Docker). It complements the existing harness, business-solutions, and ensemble-aggregation docs - here the focus is specifically the eval frameworks and repos themselves. Every license below is verified; tools that are not fully free/local/permissive are flagged explicitly.

### What an eval is (plain language)

An **eval** (evaluation) is an automated test for an AI model's output. A normal software test checks `2 + 2 == 4` - an exact, deterministic answer. But AI outputs are free-form text, so "is this answer good?" is fuzzy. Evals are the techniques that turn that fuzzy question into a repeatable score or pass/fail. Jargon decoded:

- **LLM** - Large Language Model (the AI that writes text). **Local model** - one running on your own machine via LM Studio, so there is no per-call fee and no data leaves your computer.
- **LLM-as-judge** - using one AI model to grade another AI's answer against a rubric. Our FreeStack lab already does this (a "jury" of local models voting via litellm + a JSON schema).
- **Rubric / criteria grading** - a written checklist (e.g. "is the tone polite? is it factually correct? does it follow policy?") the judge scores against.
- **G-Eval** - a popular method where the judge thinks step-by-step (chain-of-thought) against your custom criteria before scoring. Good for any free-form output.
- **Benchmark** - a fixed set of questions with known correct answers (e.g. MMLU, GSM8K) used to measure raw model capability so numbers are comparable across models.
- **RAG** - Retrieval-Augmented Generation: the app fetches documents and the model answers from them. RAG evals check the answer is **grounded** (actually based on the retrieved docs) and not **hallucinated** (made up).
- **Red-team** - deliberately attacking the model (jailbreaks, prompt injection, tricking it into leaking data) to find safety holes before customers do.
- **CI gate** - an automated check in your build pipeline that **blocks a release** if quality drops below a threshold.
- **Tracing / observability** - recording every call (inputs, outputs, cost, latency) so you can debug and watch quality over time in production.

### How evals map to business needs

| Business question | Eval type | What it gives you |
|---|---|---|
| "Did our chatbot get worse after we changed the prompt or model?" | Output-quality + CI regression | A build that blocks bad deploys |
| "Which free local model is good enough to ship?" | Capability benchmarks + pairwise | Comparable numbers to justify a choice |
| "Is our doc-chat answering from the source, not making things up?" | RAG evaluation | Proof of grounding before customer rollout |
| "Can a malicious user jailbreak us or leak data?" | Safety / red-team | A security report for audit/compliance |
| "Can our agent actually complete the multi-step task?" | Agent / coding benchmarks | Pass-rate on realistic tasks |
| "Is quality drifting in production right now?" | Observability + eval | A live dashboard and a quality loop |

---

### 1. LLM-as-judge and output-quality evals

Grade free-form outputs (tone, correctness, policy) with rubrics, G-Eval, pairwise comparison, and custom metrics.

| Tool | What it measures | License | Local-free | Business use | Install |
|---|---|---|---|---|---|
| **DeepEval** | 30+ metrics: G-Eval custom rubrics, relevancy, faithfulness, hallucination, toxicity, bias, plus agentic/conversational; pairwise | Apache-2.0 | Yes - judge model swappable to LM Studio/Ollama via base_url or litellm string. Set `DEEPEVAL_TELEMETRY_OPT_OUT=YES` for air-gapped | Reusable regression suite for a support chatbot; CI blocks a deploy if a rubric score drops | `pip install -U deepeval` |
| **promptfoo** | YAML-driven assertions: contains/regex/JSON-schema/cost/latency plus llm-rubric, g-eval, factuality, pairwise; side-by-side matrix UI | MIT | Yes - first-class Ollama/llama.cpp/OpenAI-compatible; judge also local | Grade 5 candidate prompts against 50 test cases, produce a shareable matrix for stakeholders | `npm install -g promptfoo` |
| **Verdict** | Build compound judges: ensembles, debate, verification, aggregation (mean/majority) - more reliable than one LLM call | MIT | Yes - litellm is a core dep, so any local endpoint works; examples target OpenAI so you wire the model string yourself | Harden the FreeStack jury into a principled, reusable ensemble judge | `pip install verdict` |
| **Inspect AI** | Production framework: datasets, solvers, model-graded scorers, bootstrap confidence intervals, pass/fail gates, log viewer | MIT (UK AISI) | Yes - many providers incl. vLLM/Ollama/OpenAI-compatible; grader configurable | Auditable eval standard for a regulated product, with statistical CIs as due-diligence evidence | `pip install inspect-ai` |
| **RAGAS** | Reference-free metrics incl. AspectCritic and rubric scoring; also synthetic test-set generation | Apache-2.0 | Yes - judge LLM + embeddings pluggable to local | Best when outputs are retrieval-grounded (see RAG section) | `pip install ragas` |
| **Langfuse** | Trace- and dataset-level eval with built-in judge templates and custom rubrics; dashboards over time | MIT core (avoid `ee/`) | Yes - self-host free; judge to any OpenAI-compatible endpoint | Continuous production quality monitoring with a closed feedback loop | `docker compose up` |
| **OpenAI Evals** | Original framework + registry; YAML model-graded templates (fact, closed-QA) | MIT code (some bundled datasets differ) | **Partial** - built around OpenAI API; local works via OPENAI_BASE_URL but needs manual wiring, lower maintenance velocity | Adapt a known YAML grader template as a starting point | `pip install evals` |

Honest notes: DeepEval and promptfoo are the strongest fits for assembling many reusable evals fast. Verdict is newer/smaller and most useful when judge reliability itself is the problem. Inspect AI is the heaviest but most durable for high-stakes work. OpenAI Evals is somewhat legacy - prefer the others for fresh local evals.

### 2. Capability benchmarks (MMLU, GSM8K, IFEval, BBH) and their runners

Standard academic task sets with known answers, for comparable capability numbers.

| Tool | What it measures | License | Local-free | Business use | Install |
|---|---|---|---|---|---|
| **lm-evaluation-harness** | The de-facto runner: 60+ benchmarks (MMLU, GSM8K, ARC, BBH, IFEval, GPQA); standardized few-shot | MIT | Yes - `--model local-completions`/`local-chat-completions` at LM Studio. Prefer generative/chat tasks when logprobs unavailable | Regression-gate every fine-tune/model in CI; produces canonical numbers | `pip install lm-eval` |
| **lighteval (HF)** | 1000s of tasks (MMLU, MMLU-Pro, GSM8K, BBH, IFEval, MATH); cleaner multi-backend | MIT | Yes - explicit LiteLLM backend at any OpenAI-compatible endpoint | Scriptable internal benchmark library on the same litellm path as the jury | `pip install lighteval[litellm]` |
| **OpenAI simple-evals** | Readable reference impls (MMLU, MATH, GPQA, DROP, HumanEval) | MIT | **Partial** - OpenAI/Anthropic samplers; point base_url at LM Studio with a small edit, no turnkey flag | A tiny, auditable codebase to fork into a bespoke branded eval | clone repo, run as scripts |
| **OpenCompass** | 100+ datasets, strong Chinese/bilingual (CMMLU, C-Eval) | Apache-2.0 | Yes - OpenAI-interface endpoint or local HF | Certify a model on bilingual benchmarks before APAC rollout | `pip install opencompass` |
| **Stanford HELM** | Holistic: accuracy + calibration, robustness, fairness, bias, toxicity, efficiency | Apache-2.0 | **Partial** - local via OpenAI-compatible/vLLM wiring; heavier, provider-account-oriented setup | Governance/model-card deliverable beyond a single accuracy number | `pip install crfm-helm` |
| **EvalScope (ModelScope)** | Capability benchmarks **plus** throughput/latency stress-testing in one tool | Apache-2.0 | Yes - `--eval-type openai_api --api-url` at LM Studio | Quote both quality and tokens/sec to a customer from one run | `pip install evalscope` |
| **Evalchemy** | Wraps lm-eval, adds MTBench/AlpacaEval/MixEval under one CLI | **UNKNOWN - no LICENSE file (all rights reserved)** | Partial (inherits lm-eval) | - | clone + `pip install -e .` |

Honest notes: lm-eval is the best default; lighteval slots cleanest into the existing litellm pattern. **Avoid Evalchemy for commercial use** - it has no license file, so its own code defaults to all-rights-reserved; use upstream lm-eval (MIT) instead, or open an issue requesting an OSS license first. For all runners, the benchmark **datasets** carry their own per-task licenses - the runner's license does not cover redistributing data.

### 3. Agent and coding task benchmarks

Multi-step agent and coding evals - does the agent actually complete the task?

| Tool | What it measures | License | Local-free | Business use | Install |
|---|---|---|---|---|---|
| **SWE-bench** | Can an agent resolve real GitHub issues so hidden tests pass; Docker-isolated; Verified/Lite subsets | MIT | Yes - harness only scores; pair with an agent pointed at LM Studio | Regression-gate a code-fixing/copilot product on real bug-fix capability | `pip install swebench` (Docker) |
| **SWE-smith** | Task-instance **factory**: generate unlimited SWE-bench-style tasks from any Python repo | MIT | Yes - designed to train/eval local LMs | Mint a private coding-eval set from your own codebase without exposing it | clone + `pip install -e .` (Docker) |
| **Terminal-Bench** | End-to-end agent competence at sandboxed terminal tasks (compile, configure, debug) | Apache-2.0 | Yes - LiteLLM-integrated; tasks run in local Docker | Validate a DevOps/SRE or installer agent against repeatable shell tasks | `pip install terminal-bench` (Docker) |
| **Inspect AI + inspect_evals** | Agent framework plus a library packaging GAIA, SWE-bench etc. as ready tasks under one runner | MIT | Yes - OpenAI-compatible/Ollama provider; sandboxing in local Docker | Standardize all business evals on one harness; wire your jury as a scorer | `pip install inspect-ai inspect-evals` |
| **tau-bench / tau2-bench** | Tool-agent-user customer-service interactions (retail, airline); scores DB end-state + policy; pass^k reliability | MIT | Yes - LiteLLM; both agent and simulated-user models can be local | Evaluate a support/booking agent against YOUR policies and API schemas | clone tau2-bench + `pip install -e .` |
| **GAIA** | 466 real-world multi-step tool-use questions; exact-match graded | **Dataset gated/no-reshare; Gaia2 CC-BY-4.0; runner code MIT/Apache** | Yes for the model; web tasks may need live internet/search backends | Internal yardstick for a do-anything assistant, then build your own private set | `pip install inspect-evals` |
| **WebArena / VisualWebArena** | Autonomous web-agent tasks in self-hosted site clones; programmatic success checks | Apache-2.0 | **Partial** - model is pluggable but you must host several-GB Docker site images | Stress-test a web-automation/RPA product against reproducible sites | clone + `pip install -r requirements.txt` |

Honest notes: Inspect AI + inspect_evals is the lowest-friction entry and the integration backbone for this category. tau2-bench is the closest fit for non-coding business agents (support/ops) - prefer it over tau-bench v1. **GAIA's original dataset is gated with a no-reshare clause** - use it internally only and do not redistribute or train on the test split; build your own private GAIA-style set for product gating. WebArena is the heaviest infra - worth it only if browser automation is your product line. "Free" for tau-bench/GAIA assumes you also run the simulated-user model locally.

### 4. RAG evaluation

Retrieval-augmented QA quality: faithfulness, context relevancy/precision/recall, hallucination/groundedness.

| Tool | What it measures | License | Local-free | Business use | Install |
|---|---|---|---|---|---|
| **RAGAS** | Canonical RAG suite: faithfulness, answer/context relevancy, context precision/recall; synthetic test-set generation | Apache-2.0 | Yes - LLM + embeddings fully decoupled; LM Studio/Ollama for both | Regression-gate a support RAG bot; bootstrap a golden set from product docs | `pip install ragas` |
| **DeepEval** | Full RAG set (faithfulness, contextual precision/recall/relevancy) + dedicated hallucination metric, as pytest assertions | Apache-2.0 | Yes - Ollama, LiteLLM, OpenAI-compatible; some metrics run with no LLM call | RAG assertions as pytest tests that gate every commit | `pip install deepeval` |
| **TruLens** | The "RAG Triad" - context relevance, groundedness, answer relevance - to localize whether retrieval or generation failed; OTel tracing | MIT | Yes - ships a LiteLLM provider; point api_base at LM Studio | Instrument a production RAG app; triage hallucination vs bad retrieval | `pip install trulens trulens-providers-litellm` |
| **continuous-eval (Relari)** | Mixes deterministic (ROUGE/precision-recall on chunks), semantic, and LLM-based metrics to save judge calls | Apache-2.0 | **Partial** - deterministic metrics free/offline; local LLM judge supported but less documented | Cost-control pipeline: deterministic on every run, reserve the local judge for generation checks | `pip install continuous-eval` |
| **Arize Phoenix** | Pre-built RAG judges (hallucination, QA correctness, retrieval relevance) + OTel tracing in a local UI | **MIXED - core + evals package are Elastic License 2.0 (ELv2); only client/otel subpackages Apache-2.0** | Yes - runs locally via pip/Docker; evaluators support LM Studio/Ollama | Local Dockerized observability + eval dashboard during development | `pip install arize-phoenix` |

Honest notes: RAGAS is the best first pick and maps cleanly onto our jury (it decomposes answers into claims and judges each) - but token-hungry metrics can be noisy on small local judges, so cross-check against the jury, and pin v0.2+ (the API differs from old v0.1 tutorials). continuous-eval's deterministic retrieval metrics are a genuine differentiator but **its maintenance cadence has slowed** (no 2025-2026 releases at last check) - treat it as a vendorable toolbox, not a long-term dependency. **Phoenix's ELv2 covers the evals package you import** - fine for internal use and embedding in products, but you may **not** offer Phoenix itself as a hosted service. If you need resale-as-a-service, prefer Langfuse or Opik (below).

### 5. Safety, red-team and robustness

Find jailbreaks, prompt injection, data leakage, toxicity before customers do.

| Tool | What it measures | License | Local-free | Business use | Install |
|---|---|---|---|---|---|
| **garak (NVIDIA)** | 50+ probes: jailbreaks, prompt injection, PII leakage, toxicity, malware, encoding evasions; scored report | Apache-2.0 | Yes - `openai.OpenAICompatible`/litellm generator. **Gotcha: set the LM Studio uri in a YAML config, not CLI - CLI uri override is silently ignored** | Nightly pre-ship security gate; attach the report to a compliance package | `pip install garak` |
| **PyRIT (Microsoft)** | Multi-turn attack orchestration (Crescendo, TAP, PAIR, Skeleton Key); attacker-LLM-vs-target loops | MIT | Yes - attacker + scorer as OpenAIChatTarget at LM Studio; notebook-driven | Automated multi-turn jailbreak regression suite for an agentic product | `pip install pyrit` |
| **promptfoo (red-team mode)** | Auto-generated adversarial cases for OWASP LLM Top 10, jailbreaks, injection, PII; severity report | MIT | Yes - Ollama/OpenAI-compatible providers; runs locally | Lowest-friction CI red-team gate; swap the provider block across client products | `npx promptfoo@latest redteam init` |
| **DeepTeam (Confident AI)** | 120+ vulnerabilities, 20+ attack vectors; maps results to OWASP/NIST AI RMF/MITRE ATLAS | Apache-2.0 | Yes - `deepteam set-local-model --base-url` for LM Studio; Ollama via `deepeval set-ollama` | Compliance-framed red-team report mapped to standards for audit/sales | `pip install deepteam` |
| **Giskard (giskard-oss)** | LLM `scan` (hallucination, toxicity, injection, bias, disclosure) + RAGET RAG test-set generation; HTML report | Apache-2.0 | **Partial** - judge via litellm model strings to LM Studio; v3 is a beta rewrite, verify a detector exists in your version | RAG-product QA: synthesize a test set, scan for regressions per client | `pip install giskard` |
| **LLM Guard (Protect AI)** | 35+ input/output scanners (injection, jailbreak, toxicity, PII, secrets) as deterministic detectors | MIT | Yes - runs locally on HF models (one-time download); also a Docker API server | Deterministic detector layer for a guardrail proxy; reuse to score attack outputs | `pip install llm-guard` |

Honest notes: garak (broad canned scan) and PyRIT (programmable multi-turn orchestration) complement each other. DeepTeam slots cleanest into a stack already using DeepEval and has the best compliance-mapping story (note: Ollama setup uses `deepeval set-ollama`, not `deepteam`). Giskard is strongest on bias/RAG but **its v3 rewrite is beta with feature drift from v2 - pin a version and verify detectors**. LLM Guard is not a red-team generator but a fast local detector layer, so you avoid burning judge calls on obvious PII/toxicity hits; pair with Microsoft Presidio (MIT) for GDPR/HIPAA-grade redaction.

### 6. Product CI, regression and observability

Catch quality regressions in YOUR app when prompts/models change; tracing + eval that gates a CI build.

| Tool | What it measures | License | Local-free | Business use | Install |
|---|---|---|---|---|---|
| **promptfoo** | Versioned YAML test cases + assertions; diffs outputs across prompts/models and fails the build on regressions; junit/json/html output | MIT (OpenAI-owned 2026; license unaffected) | Yes - `openai:chat:<model>` + OPENAI_BASE_URL at LM Studio; results stay local | "Prompt regression suite" deliverable: GitHub Action blocks a PR on score drop | `npm install -g promptfoo` |
| **deepeval** | "Pytest for LLMs": 50+ metrics as assertions; `deepeval test run` as a CI gate; dataset snapshots, span-level evals | Apache-2.0 | Yes - DeepEvalBaseLLM/OpenAI-compatible to LM Studio; some metrics no-LLM | Ship a pytest eval suite into a client's existing CI - no new platform to learn | `pip install deepeval` |
| **Langfuse** | Tracing + datasets + prompt versioning + judge evaluators + experiment comparison to catch drift in CI and production | MIT core (`/ee` opt-in, dormant without a key) | Yes - self-host via docker compose; judge to LM Studio | Auditable trace + cost layer plus a regression dashboard on model bumps | `docker compose up` |
| **Arize Phoenix** | OTel-native tracing + eval templates + dataset experiments; strong retrieval debugging | **Elastic License 2.0 (ELv2) incl. the evals package - source-available, no managed-service** | Yes - local via pip/Docker; eval models at LM Studio | Debug/regression-test a client's RAG retrieval during dev; embed on-prem (not resell-as-service) | `pip install arize-phoenix` |
| **autoevals (Braintrust)** | Ready-made scorers (Factuality, ClosedQA, Battle/pairwise, JSON-diff, similarity) for your own CI loop | MIT | **Partial** - heuristic scorers local; LLM-graded honor OPENAI_BASE_URL - set it to LM Studio or it falls back to Braintrust/OpenAI | Embed a couple of scorers in a tiny CI script - minimal dependency footprint | `pip install autoevals` |
| **Ragas** | RAG metrics in CI; integrates into pytest, Langfuse, Phoenix | Apache-2.0 (now vibrantlabsai/ragas) | Yes - any LangChain LLM + local embeddings | "Prove the RAG didn't get worse" gate on knowledge-base or model updates | `pip install ragas` |
| **Opik (Comet)** | Self-hostable tracing + eval + monitoring; experiments, prompt versioning, pytest integration | Apache-2.0 | Yes - self-host via docker compose; judges at LM Studio/Ollama | Resellable self-hosted Braintrust/LangSmith alternative - you **may** run it as a service | `pip install opik` |
| **Evidently** | Explicit Test/TestSuite with pass/fail thresholds + data/embedding **drift** detection; CI gate and scheduled monitor | Apache-2.0 | Yes - local/free; judge descriptors to LM Studio; many checks no-LLM | Fail CI on regressions AND run scheduled drift reports auditors accept | `pip install evidently` |

Honest notes: promptfoo (YAML/CLI) and deepeval (code/pytest) are the two primary gates - pick by whether the team lives in config or in pytest. For an all-in-one tracing+eval+monitor platform you can **resell as a hosted service**, prefer **Opik or Langfuse (permissive)** over **Phoenix (ELv2 forbids managed-service resale, and that clause now covers its evals package)**. Evidently uniquely covers input drift, useful when the regression source is the data, not the prompt. promptfoo is now OpenAI-owned (2026) but remains MIT.

---

### Recommended free-local starter stack

Adopt in this order - each step adds a capability without throwing away the last:

1. **promptfoo** (Product CI / output-quality / red-team, MIT) - start here. One YAML file gives rubric grading, model-comparison matrices, CI gates, and red-team mode, with the best out-of-the-box LM Studio support. Fastest path to your first reusable eval.
2. **DeepEval** (LLM-as-judge / RAG / CI, Apache-2.0) - add the code-first, pytest-native side. Largest reusable metric library (G-Eval, faithfulness, hallucination) that becomes CI assertions. Set `DEEPEVAL_TELEMETRY_OPT_OUT=YES` for an air-gapped run.
3. **RAGAS** (RAG, Apache-2.0) - add the moment a product retrieves from a knowledge base. Faithfulness/context metrics plus synthetic test-set generation to bootstrap golden sets. Pin v0.2+.
4. **lm-evaluation-harness** (Capability benchmarks, MIT) - add to produce the canonical MMLU/GSM8K/IFEval numbers stakeholders recognize and to regression-gate fine-tunes. Prefer chat/generative tasks given LM Studio logprobs variance.
5. **garak** (Safety/red-team, Apache-2.0) - add a pre-ship security scan. Pure CLI, batch report. Remember the YAML-config gotcha for the LM Studio uri.
6. **Langfuse** (Observability, MIT core) - add last, once evals exist and you need persistence, a UI, drift tracking over time, and a loop that feeds low-scoring production traces back into eval datasets. Mind the Postgres + ClickHouse + Redis footprint for production, and stay out of `/ee`.

This sequence spans all six categories, every entry is free + local + permissively licensed (MIT/Apache-2.0), and difficulty rises gradually from one YAML file to a self-hosted platform.

### How this connects to our FreeStack local jury

We already have an LLM-as-judge jury (litellm + json_schema over LM Studio, with ensemble voting). The catalog tools **extend** rather than replace it - the jury is the judging brain; these tools are the test-runners, metric libraries, and dashboards around it.

- **Wrap the jury as a custom metric/scorer (keep it, gain a harness):** DeepEval (custom `DeepEvalBaseLLM`), promptfoo (python/javascript or `llm-rubric` assertion calling the jury endpoint), Inspect AI / inspect_evals (custom scorer), and PyRIT/DeepTeam (custom scorers) all let the jury be the grader inside a bigger, reusable suite. This is the highest-leverage move: the existing jury becomes reusable across every product eval.
- **Harden the jury itself (upgrade, not replace):** **Verdict** most directly extends the jury and the Ensemble-Aggregation doc - it operationalizes ensemble/aggregation/debate schemes specifically for judges, turning ad-hoc 2-3 model voting into a declared, reusable, reliability-hardened pipeline. Both share the litellm dependency.
- **Reuse the same litellm -> LM Studio path:** lighteval, RAGAS, TruLens, tau2-bench, and Terminal-Bench all route through litellm, so they slot into the existing client config with no new plumbing.
- **Add persistence and a UI the jury lacks:** Langfuse and Opik (both permissive, self-hostable) give the jury's scores a home - traces, dashboards, dataset experiments, and a production quality loop. Phoenix does this too but is ELv2 (no resale-as-service).
- **Where deterministic detectors replace a judge call:** LLM Guard, continuous-eval's deterministic metrics, and promptfoo's exact/regex/JSON-schema assertions catch obvious failures (PII, toxicity, format) without invoking the jury - cheaper and faster, reserve the jury for genuinely fuzzy quality calls.

Nothing here needs to **replace** the jury. The clearest single upgrade is Verdict (better judge), and the clearest single multiplier is wrapping the jury as a DeepEval/promptfoo metric so it powers many reusable evals at once.