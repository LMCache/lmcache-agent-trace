This repo is meant to run a series of experiments to collect trace data from AI agents. And further use the data to analyze the performance of the agents, following the idea from https://github.com/LMCache/LMCache/issues/1826.

**Important note**: If the agent is powered by an OPENAI model, the traces will be automatically tracked from the [OpenAI API Dashboard](https://platform.openai.com/logs). Collecting traces should be easy.

A simple vibe-coded agent trace prefix [analyzer](https://v0-llm-agent-dashboard.vercel.app/).

## On-going Checklist

**Status Legend:**
✅ Done, 🔄 Ongoing, ⏳ Wait to be taken, ❌ Failed

| Project Name | Status | Notes |
|--------------|--------|-------|
| [Terminus 1&2](https://www.tbench.ai/terminus) | ✅ | From [terminal bench](https://www.tbench.ai/) and [harbor](https://harborframework.com/docs/running-tbench) project, source code open; Basically the agent context management is to append new round conversation on all the previous chat history. Because of this nature, prefix and non-prefix cache hit rates should be stricly the same. Each step, it plans and executes multiple functions in sequential order. Easy to collect traces. |
| [Mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) | ✅ | Running this agent with the [terminal bench](https://www.tbench.ai/) and [harbor](https://harborframework.com/docs/running-tbench), the only difference between Terminus 1&2 is that Mini-swe-agent plans and executes one bash command(or multiple ones with '&&' but rarely happens) at one time, while Terminus 1&2 plans multiple commands and executes them in sequential order. So the context reuse pattern is very similar to Terminus 1&2. |
| [Claude Code Agent](https://pierce.dev/notes/under-the-hood-of-claude-code) | 🔄 | References: https://pierce.dev/notes/under-the-hood-of-claude-code, https://pierce.dev/notes/a-deep-dive-on-agent-sandboxes, https://medium.com/@outsightai/peeking-under-the-hood-of-claude-code-70f5a94a9a62, https://platform.claude.com/docs/en/build-with-claude/prompt-caching#structuring-your-prompt; Planning to release a huggingface blog on 11/26/2025|
| [MetaGPT](https://github.com/geekan/MetaGPT) | 🔄 | The Multi-Agent Framework: First AI Software Company, Towards Natural Language Programming |
| [OpenHands](https://github.com/All-Hands-AI/OpenHands) | 🔄 | OpenHands: Code Less, Make More. (formerly OpenDevin), a platform for software development agents powered by AI |
| [GPT Pilot](https://github.com/Pythagora-io/gpt-pilot) | 🔄 | GPT Pilot is the core technology for the Pythagora VS Code extension that aims to provide the first real AI developer companion |
| [Aider](https://github.com/Aider-AI/aider) | 🔄 | aider is AI pair programming in your terminal |
| [Devika](https://github.com/stitionai/devika) | ⏳ | Devika is an Agentic AI Software Engineer that can understand high-level human instructions, break them down into steps, research relevant information, and write code to achieve the given objective |
| [RepoAgent](https://github.com/OpenBMB/RepoAgent) | 🔄 | An LLM-powered repository agent designed to assist developers and teams in generating documentation and understanding repositories quickly |
| [DSPy](https://github.com/stanfordnlp/dspy) | 🔄 | The framework for programming—not prompting—foundation models |
| [ThinkGPT](https://github.com/alaeddine-13/thinkgpt) | 🔄 | Agent techniques to augment your LLM and push it beyond its limits |
| [PyCodeAGI](https://github.com/chakkaradeep/pyCodeAGI) | ⏳ | A small AGI experiment to generate a Python app given what app the user wants to build |
| [SuperAGI](https://github.com/TransformerOptimus/SuperAGI) | ⏳ | SuperAGI - A dev-first open source autonomous AI agent framework |
| [MemGPT(letta)](https://github.com/letta-ai/letta) | ⏳ | Letta is the platform for building stateful agents: open AI with advanced memory that can learn and self-improve over time. |

More agents to be added from the [Awesome Agents](https://github.com/kyrolabs/awesome-agents?tab=readme-ov-file#software-development) repository, specifically focusing on the **Software Development** category. If the list is drained, use the [backup list](https://github.com/e2b-dev/awesome-ai-agents?tab=readme-ov-file). Contact [Kobe](https://github.com/kobe0938) for any questions.

References:
* https://github.com/MLSysOps/Agent-Benchmarking-LMCache (open_deep_research and langmanus)
* https://www.anthropic.com/engineering/building-effective-agents
* https://www.arxiv.org/pdf/2510.04618 (Agentic Context Engineering)
* https://arxiv.org/pdf/2511.02230 (CONTINUUM)
* https://arxiv.org/pdf/2410.02506 (AgentPrune)
* https://arxiv.org/pdf/2510.12872 (KVCOMM)