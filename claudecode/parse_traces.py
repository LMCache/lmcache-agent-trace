#!/usr/bin/env python3
"""Parse traces.jsonl into a simplified JSONL where each line is one LLM call."""

import json
import argparse
import uuid


def strip_cache_control(obj):
    """Recursively remove all 'cache_control' keys from dicts/lists."""
    if isinstance(obj, dict):
        return {k: strip_cache_control(v) for k, v in obj.items() if k != "cache_control"}
    if isinstance(obj, list):
        return [strip_cache_control(item) for item in obj]
    return obj


def build_input(raw_request):
    """Concatenate system, tools, and messages from raw_request into a single string.

    Sections are separated by tags like <system>, <tools>, <messages>,
    each on its own line. Items within a section are separated by newlines.
    """
    parts = []

    system = raw_request.get("system", [])
    if system:
        parts.append("<system>")
        for item in system:
            parts.append(json.dumps(strip_cache_control(item)))
        parts.append("")

    tools = raw_request.get("tools", [])
    if tools:
        parts.append("<tools>")
        for item in tools:
            parts.append(json.dumps(strip_cache_control(item)))
        parts.append("")

    messages = raw_request.get("messages", [])
    if messages:
        parts.append("<messages>")
        for item in messages:
            parts.append(json.dumps(strip_cache_control(item)))
        parts.append("")

    return "\n".join(parts)


def parse_trace_line(line_data):
    """Extract relevant fields from a single trace entry."""
    response = line_data.get("response", {})
    choices = response.get("choices", [])
    message = choices[0]["message"] if choices else {}
    raw_request = line_data.get("raw_request", {})

    return {
        "timestamp": line_data.get("start_time"),
        "end_time": line_data.get("end_time"),
        "input_tokens": line_data.get("input_tokens"),
        "output_tokens": line_data.get("output_tokens"),
        "input": build_input(raw_request),
        "output": json.dumps(message),
        "session_id": "session_dummy_001",
    }


def main():
    parser = argparse.ArgumentParser(description="Parse traces.jsonl into per-LLM-call JSONL")
    parser.add_argument("--input", default="traces.jsonl", help="Input traces file")
    parser.add_argument("--output", default="parsed_traces.jsonl", help="Output parsed file")
    args = parser.parse_args()

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            parsed = parse_trace_line(data)
            fout.write(json.dumps(parsed) + "\n")

    print(f"Parsed {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
