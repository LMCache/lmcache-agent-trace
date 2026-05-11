#!/usr/bin/env python3
"""Generate a synthetic coordination-agent trace.

The trace is deliberately hand-authored and deterministic. It models the shape
of a public coordination workflow without reading private logs, transcripts, or
environment state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


STATIC_SYSTEM = """[System]
You are a coordination agent for a public open-source project.
You only use public artifacts, explicit receipts, and reproducible checks.
You must separate observation, inference, action, and follow-up.
Never include private transcripts, credentials, personal data, or hidden notes.
"""


SCENARIOS = [
    {
        "name": "public_issue_followup",
        "task": "Review a public GitHub issue after a collaborator replies.",
        "public_artifacts": [
            "GitHub issue URL: https://github.com/example-org/example-project/issues/64",
            "Project status page: https://example.org/project/status",
            "Previous public comment: clarify owner-of-record before implementation",
        ],
        "checks": [
            "Read the latest public issue comments.",
            "Confirm the reply requests protocol clarification rather than code.",
            "Check that no private support channel is referenced.",
        ],
        "action": "Post one concise public clarification and update the watchlist.",
        "handoff": "Wait for maintainer confirmation before opening a PR.",
    },
    {
        "name": "resolution_nudge_measurement",
        "task": "Measure whether a public market-resolution nudge attracted attention.",
        "public_artifacts": [
            "Market URL: https://example.com/markets/public-resolution-example",
            "Pre-comment snapshot: traders=43, volume=4610.58",
            "Public comment label: resolution-status nudge",
        ],
        "checks": [
            "Fetch the public market page after the six-hour window.",
            "Compare trader and volume deltas against the pre-comment snapshot.",
            "Do not post a second top-level comment without a reply or exception.",
        ],
        "action": "Record a post-window metric receipt with delta traders and volume.",
        "handoff": "Schedule the 24-hour measurement window.",
    },
    {
        "name": "blog_receipt_review",
        "task": "Review a public blog draft for provenance and uncertainty receipts.",
        "public_artifacts": [
            "Draft title: Small Samples Need Assumption Receipts",
            "Public source card: small-N forecasting note",
            "Site policy: posts need owner, inference, uncertainty, and falsifier fields",
        ],
        "checks": [
            "Scan front matter for author and owner fields.",
            "Flag claims that cite anecdotal examples without source caveats.",
            "Keep lint advisory-only so publication is not blocked by old posts.",
        ],
        "action": "Publish the post only after adding source caveats and an owner receipt.",
        "handoff": "Add the lint finding to the daily culture queue.",
    },
]


def session_id(name: str) -> str:
    return hashlib.md5(name.encode("utf-8")).hexdigest()


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    timestamp = 0

    for scenario in SCENARIOS:
        sid = session_id(scenario["name"])
        memory = ""

        steps = [
            (
                "Orient",
                "Identify the task, public artifacts, and private-data boundary.",
                (
                    f"Task: {scenario['task']}\n"
                    "Boundary: use public artifacts only; no private logs or secrets."
                ),
            ),
            (
                "Source Map",
                "Build a source map from the public artifacts.",
                "Source map:\n- " + "\n- ".join(scenario["public_artifacts"]),
            ),
            (
                "Policy Check",
                "Apply the coordination rules before taking action.",
                "Checks:\n- " + "\n- ".join(scenario["checks"]),
            ),
            (
                "Receipt",
                "Write the minimal receipt for the safe action.",
                f"Action receipt: {scenario['action']}",
            ),
            (
                "Handoff",
                "Name the next owner-visible follow-up.",
                f"Next action: {scenario['handoff']}",
            ),
        ]

        for step_name, instruction, output in steps:
            input_text = (
                f"{STATIC_SYSTEM}\n"
                f"[Workflow]\nName: {scenario['name']}\nTask: {scenario['task']}\n"
                f"{memory}\n"
                f"[User]\nStep: {step_name}\n{instruction}"
            )
            rows.append(
                {
                    "timestamp": timestamp,
                    "input": input_text,
                    "output": output,
                    "session_id": sid,
                }
            )
            memory += f"\n[{step_name} Output]\n{output}\n"
            timestamp += 120

        timestamp += 600

    return rows


def main() -> None:
    out_path = Path(__file__).with_name("trace.jsonl")
    rows = build_rows()
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
