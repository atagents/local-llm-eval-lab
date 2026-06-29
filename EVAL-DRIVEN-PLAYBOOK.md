# Eval-Driven Engineering — a portable playbook (mentor reference)

A reusable reference for building AI systems by **measurement, not vibes**. Distilled from the
FreeStack lab and proven on a real fine-tune loop (see the case study at the end). Written so a
Claude Code session on another machine (or a human) can apply it to *any* agent / model / prompt
work — PulseSales, Hermes, AutoEvalsBots, or a new project.

## The one rule
**Don't trust a change — measure it on a held-out test, and ship only what survives the statistics.**
"It feels better" is how you ship a regression. Every claim of improvement needs a before, an after,
and a confidence interval.

## The loop
1. **Define the metric that should move.** What is this change *supposed* to improve? If you can't
   name a number that would move, the change is premature. (For an agent: task-success rate, the
   specific failure you're fixing, safety/ASR, latency — not a vague "quality".)
2. **Build a held-out test.** Cases the system never trains/tunes on. Make them **hard** — easy cases
   ceiling out and can't distinguish anything. Keep the test set secret from the thing under test
   (no leakage).
3. **Baseline.** Measure the current system on the test. This is your "before".
4. **Change one thing.** New prompt, fine-tune, model, tool.
5. **Re-measure.** Same test, same judge.
6. **Decide on the lower bound, not the point estimate.** Use a paired test + CI. `+0.1` is usually
   noise. Ship only if the CI lower bound clears zero (IMPROVE), hold if FLAT, revert if REGRESS.
7. **Gate it.** Pin the baseline in CI; every future change must pass or it doesn't ship.

## Hard-won lessons (each cost a real experiment)
- **The mean is often the wrong metric.** If the average is near the ceiling, a real improvement
  won't move it. Measure the **failure-rate** (% of cases scoring ≤2), the **floor/variance**, and
  the **specific thing** you changed (e.g., "does it hold character with no prompt").
- **Ceiling effect is real.** Easy test + lenient judge → everything scores 5.0 → useless. A strict
  rubric ("a 5 is RARE, demands all of X/Y/Z") + hard cases makes scores spread so models become
  distinguishable.
- **Judge ≠ subject.** Don't let a model judge itself or its own family (self-preference bias). Use a
  neutral judge, or a **jury of different families** with median aggregation.
- **Fine-tuning can make things WORSE.** Twice, fine-tuning on cheap auto-generated data regressed
  (overfit to a single trait, lost the rest). The eval caught it; nothing bad shipped. **The
  bottleneck is almost always data quality, not hyperparameters** — tweaking epochs/mix on weak data
  doesn't help.
- **Levers, ranked by measured impact:** *data quality > model/brain choice > prompt & agent
  engineering.* Prompt and agent-wrapper work plateaus fast; a stronger brain or genuinely better
  data is what moves the number.
- **Generate training/eval data with a frontier model + strict verification**, not a weak local one.
  Pattern that worked: fan out generation across scenario buckets → an adversarial reviewer keeps only
  excellent, non-duplicate, non-leaky examples → dedup. Cheap synthetic data is worse than none.
- **Speed and cost are real eval dimensions** for anything user-facing — a model that scores high but
  runs at 3 tok/s fails the product regardless of quality.
- **Report honestly.** A measured "no" (this didn't help / it regressed) is a *success* of the method
  and a stronger result than an unverified "yes".

## Applying it to an agent project (e.g. Hermes / PulseSales)
- The "answer" isn't the only thing — score the **trajectory**: legal tool sequence, valid tool args,
  goal reached, no forbidden action. This can be **deterministic, no LLM** (cheap, exact).
- Build a small set of **gold traces** (good runs) + adversarial bad ones; check your scorer agrees
  with the labels before trusting it.
- Add a **prompt-injection gate** (honeypot sinks → deterministic attack-success-rate) and a
  **simulated-user** eval (an LLM persona drives the agent; judge the objective outcome on-policy).
- **Gate the agent in CI** against a pinned baseline of trajectory label-agreement; a PR that
  regresses the agent exits non-zero.
- Before any model/prompt swap ships: run the gate. No green, no ship.

## The minimum checklist before claiming "it's better"
- [ ] A held-out test the change never saw, hard enough to spread scores.
- [ ] The metric you measure is the thing the change should move (not a proxy near its ceiling).
- [ ] A neutral judge (or diverse jury); judge ≠ subject.
- [ ] Before and after on the *same* cases.
- [ ] A paired statistic; decision on the CI lower bound, not the point estimate.
- [ ] Failure cases inspected (not just the mean) — what still breaks?
- [ ] If it didn't move: say so, and find the real lever (usually data or model, not knobs).

## Worked proof (the companion case)
Goal: a smart, warm companion agent. Measured arc on a hard held-out test (persona-fit / voice /
failure-rate): base+prompt 3.50/2.42/4% → agent (same prompt) 3.42/2.38/17% (**FLAT** — prompt
plateaued) → fine-tune v1 2.54/.../62% (**REGRESS**, overfit to one trait) → v2 2.29/.../67% (**REGRESS**,
overfit to bland empathy) → **agent on a strong cloud brain 4.04/3.71/4% (IMPROVE, +1.29 voice, far
beyond noise)**. Conclusion proven by numbers, not opinion: quality was **brain-limited**, not
prompt-limited; cheap-synthetic fine-tunes hurt; the fix is data quality (v3: frontier-generated +
strictly verified) or a stronger brain. Two regressions were caught and never shipped. That is the
whole point.
