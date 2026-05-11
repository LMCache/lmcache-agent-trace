# Coordination Agent

## Overview

This is a small synthetic trace for a coordination agent that repeatedly reads
public artifacts, checks policy boundaries, writes receipts, and hands off the
next action.

The trace is designed to model a common multi-agent operations pattern:

1. Orient on a public task.
2. Build a source map from public artifacts.
3. Check coordination and privacy policy.
4. Write a concise action receipt.
5. Hand off the next owner-visible follow-up.

## Synthetic Disclosure

This trace is synthetic and representative. It does not contain private
transcripts, private messages, raw logs, credentials, personal data, hidden
reasoning, or environment state. Public URLs in the trace are illustrative
examples, and the responses are hand-authored to demonstrate the repeated
context pattern.

The generator is dependency-free and deterministic. It embeds all scenario text
in `generate_trace.py` and writes `trace.jsonl`; it does not read from the local
filesystem beyond writing its output file.

## Why This Is Useful for Cache Analysis

Coordination agents often reuse a stable instruction block while accumulating
task specific receipts over several calls. That creates two reuse patterns:

- Cross-session prefix reuse from the stable system and workflow framing.
- Within-session substring reuse as each step carries forward prior receipts.

This makes the trace a compact example for comparing strict prefix reuse with
broader substring-style reuse on agent operations that are not code-editing
tasks.

## Trace Details

- Sessions: 3
- LLM calls per session: 5
- Total trace entries: 15
- Timestamp format: relative seconds from trace start
- Session IDs: MD5 hex digests of the synthetic scenario names
- Format: one JSON object per line with `timestamp`, `input`, `output`, and
  `session_id`

## Reproduce the Trace

```bash
python coordination_agent/generate_trace.py
```

## Run Cache Analysis

From the repository root:

```bash
python prefix_analysis.py -i coordination_agent/trace.jsonl -o coordination_agent_result.png --tokenizer gpt2
```

The analysis command is optional. This contribution includes the trace and
generator only, not a performance claim.
