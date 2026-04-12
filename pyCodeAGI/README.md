# PyCodeAGI

## Overview

[PyCodeAGI](https://github.com/chakkaradeep/pyCodeAGI) is a minimal AGI experiment inspired by BabyAGI and AutoGPT. Given a high-level app idea (e.g., "build a calculator app"), it autonomously generates a complete Python/Streamlit application through a sequential LLM pipeline.

This trace is based on the **GPT-4 version** (`pycodeagi-gpt4.py`), which uses a 5-step chat-format pipeline with `ChatOpenAI / gpt-4`.

> **Note**: The original `pycodeagi.py` targets `text-davinci-003` (deprecated Jan 2024). The GPT-4 version cannot be run without porting due to LangChain API changes (`v0.0.139 → v0.3+`). This trace reproduces the exact prompt templates; `generate_trace.py` serves as a reference for contributors who wish to generate a real trace once the code is ported.

## Agent Architecture

PyCodeAGI uses a fixed, deterministic 5-step pipeline. Each step's *system message* accumulates all prior outputs, creating an **expanding prefix chain**:

| Step | Task | Approx. Input Tokens |
|------|------|---------------------|
| 1 | Generate app description | ~80 |
| 2 | Design architecture | ~250 |
| 3 | Design UX flow | ~600 |
| 4 | Design code flow | ~1,200 |
| 5 | Generate app code | ~2,000 |

**Key pattern**: Each step's system message is a strict superset of the previous step's system message. This creates an **expanding prefix chain** within each session — ideal for LMCache prefix caching.

## Cache Reuse Analysis

### Within-session pattern

Steps 2–5 each extend the system message with accumulated context, so the token sequence of step N's system block is a prefix of step N+1's system block. This is a textbook case for **prefix cache matching** (LMCache / vLLM).

### Cross-session pattern

All sessions share the same static system header (~40 tokens). Multiple sessions with different objectives diverge immediately after the header, so:
- Short shared prefix hit for the static header
- Longer substring hits where similar architectural patterns appear

## Trace Details

- **Sessions**: 5 (different app objectives)
- **LLM calls per session**: 5
- **Total trace entries**: 25
- **Model**: GPT-4 (chat format, `ChatOpenAI`)
- **Timestamps**: absolute Unix microseconds (int64)
- **Session IDs**: MD5 hex digest of the app objective string
- **Synthetic disclosure**: Inputs faithfully reproduce GPT-4 prompt templates; outputs are hand-crafted representative examples.

## Reproduce the Trace

```bash
python generate_trace.py
```

No API keys or external dependencies required.

## Run Cache Analysis

```bash
cd ..
# Use --tokenizer gpt2 to avoid requiring a HuggingFace auth token
python prefix_analysis.py -i pyCodeAGI/trace.jsonl -o pyCodeAGI/pycodeagi_hit_rate.png --tokenizer gpt2
```
