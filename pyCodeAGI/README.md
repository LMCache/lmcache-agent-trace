# PyCodeAGI

## Overview

[PyCodeAGI](https://github.com/chakkaradeep/pyCodeAGI) is a minimal AGI experiment inspired by BabyAGI and AutoGPT. Given a high-level app idea (e.g., "build a calculator app"), it autonomously generates a complete Python/Streamlit application through a **6-step sequential LLM pipeline**.

## Agent Architecture

PyCodeAGI uses a fixed, deterministic 6-step pipeline where each step's output is fed as context to the next:

| Step | Task | Approx. Input Tokens |
|------|------|---------------------|
| 1 | Generate app description | ~100 |
| 2 | Design architecture | ~300 |
| 3 | Design UX flow | ~650 |
| 4 | Design code flow | ~1,350 |
| 5 | Generate coding steps | ~2,050 |
| 6 | Generate app code | ~2,450 |

**Key pattern**: Each step's input is a strict superset of the previous step's input (accumulated context). This creates an **expanding prefix chain** within each session.

## Cache Reuse Analysis

### Within-session pattern

Steps 2–6 each contain the full content of all previous steps, making the prompt of step N a substring of step N+1's prompt. This pattern is ideal for **substring cache matching** (CacheBlend).

### Cross-session pattern

All sessions share the same system prompt prefix (~50 tokens). Multiple sessions with different objectives will exhibit:
- High prefix hit rate for the shared system prompt
- Potential substring hits where similar architectural descriptions appear

## Trace Details

- **Sessions**: 5 (different app objectives)
- **LLM calls per session**: 6
- **Total trace entries**: 30
- **Model simulated**: GPT-4 (chat format)
- **Note**: This is a synthetic trace generated from PyCodeAGI's prompt templates. The input structure faithfully reproduces the actual prompts; outputs are representative examples.

## Reproduce the Trace

```bash
python generate_trace.py
```

No API keys or external dependencies required. The script uses PyCodeAGI's original prompt templates with pre-written synthetic outputs.

## Run Cache Analysis

```bash
cd ..
python prefix_analysis.py -i pyCodeAGI/trace.jsonl -o pyCodeAGI/pycodeagi_hit_rate.png
```
