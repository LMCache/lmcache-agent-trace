#!/usr/bin/env python3
"""Naive parse of traces.jsonl -- keeps raw_request and message as-is (stringified)."""

import json
import argparse
from datetime import datetime


def to_unix_timestamp(time_str):
    """Convert a datetime string to a Unix timestamp (float)."""
    if not time_str:
        return None
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp()


def parse_trace_line(line_data):
    """Extract relevant fields from a single trace entry."""
    response = line_data.get("response", {})
    choices = response.get("choices", [])
    message = choices[0]["message"] if choices else {}
    raw_request = line_data.get("raw_request", {})

    return {
        "start_time": to_unix_timestamp(line_data.get("start_time")),
        "end_time": to_unix_timestamp(line_data.get("end_time")),
        "input_tokens": line_data.get("input_tokens"),
        "output_tokens": line_data.get("output_tokens"),
        "input_raw": raw_request,
        "output_raw": message,
        "session_id": "session_dummy_001",
    }


def main():
    parser = argparse.ArgumentParser(description="Naive parse of traces.jsonl into per-LLM-call JSONL")
    parser.add_argument("--input", default="traces.jsonl", help="Input traces file")
    parser.add_argument("--output", default="parsed_traces_raw_request.jsonl", help="Output parsed file")
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
