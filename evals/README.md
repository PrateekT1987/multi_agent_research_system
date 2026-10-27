# Evaluation set

`agentic_ai_v1.json` is Mars's fixed regression set. It exists so a change to a
prompt, a model pin or agent logic is measured against the same tasks instead of
being judged by one fresh run.

## What is in the set

| Kind | Runs without an LLM | Purpose |
| --- | --- | --- |
| `domain_gate` | yes | Asserts that in-domain topics are accepted and off-domain topics are refused. |
| `injection` | needs a report | An in-domain topic carrying an instruction payload; the report must not leak a credential or repeat the payload. |
| `research` | needs a report | Structural and evidence assertions: sections, canonical tracks, sources, findings, `[S#]` citations and a minimum grounding ratio. |

Gate cases protect the four-layer domain lock. Research cases protect report
shape and evidence discipline. They do not judge prose quality; that is what the
auditor and the optional continuous-evaluation judge are for.

## Running it

Validate the dataset and run every gate case:

```bash
uv run mars eval --dataset evals/agentic_ai_v1.json
```

Evaluate saved reports (the `*-<run-id>.json` files written by a run) and fail
on a regression against a baseline report:

```bash
uv run mars eval \
  --dataset evals/agentic_ai_v1.json \
  --reports research_output \
  --baseline research_output/agentic-ai-<old-run>.json
```

The command exits non-zero when a case fails or when grounding/entailment falls
by more than the tolerance, cost rises by more than 25% or findings decrease.

## Release procedure

1. Run the suite and the eval set on the current commit; save one report per
   research case (use the case topic as the run topic).
2. Change the prompt, model pin or agent logic.
3. Re-run the same topics and compare with `--baseline`.
4. If a change is intentional, update the case's `min_*` values or
   `min_grounding` in a reviewed commit and bump the dataset `version`.
5. Record the new `prompt_version` and model pins from the run manifest in the
   release notes.

A new model or prompt that cannot pass the existing cases is not ready. Do not
weaken a case to make a release pass without saying why in the commit message.
