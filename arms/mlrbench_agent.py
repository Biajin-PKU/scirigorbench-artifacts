#!/usr/bin/env python3
"""Second external arm: MLR-Agent, the reference agent shipped with MLR-Bench.

The first external arm returned nothing scoreable, and the cause was the
substrate rather than the agent: a PDF manuscript and summaries that record the
code which produced the numbers instead of the numbers. This arm is the
controlled comparison. Same repository, same ten tasks, same human hallucination
labels -- and results written as a Markdown table with a row label and a column
header, which is a record of the results as DATA.

    agent_results/experiments_and_writeups/claude/<task>/
        results/paper_<model>.md      claim side
        results/results.md            artifact side, pipe tables
        claude_code/*.py              artifact side, settings
        claude_output.json            raw trace -- excluded

If the construction reads this arm and not the other, the difference is a
statement about recording discipline, which is what column 1 exists to say.
If it reads neither, the instrument is the problem and the leaderboard cannot
be built as designed. Either answer is worth having; they are not the same
answer, and only running both distinguishes them.

⭐ Three manuscripts are generated from ONE set of artifacts (o4-mini,
claude-3-7-sonnet, gemini-2.5-pro). Holding the evidence fixed while varying the
writer isolates the generator from the run -- a within-arm control the design
did not anticipate having.

Held-out split, fixed before measurement: iclr2025_dl4c is the DEVELOPMENT task
for the Markdown path; the other nine are held out.

    python3 arms/mlrbench_agent.py --repo /tmp/mlrb/r --task iclr2025_dl4c
    python3 arms/mlrbench_agent.py --repo /tmp/mlrb/r --all
"""

import argparse
import json
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import phi  # noqa: E402

DEV_TASK = "iclr2025_dl4c"
BACKEND = "claude"
CONTRACT = pathlib.Path(__file__).resolve().parent.parent / \
    "contracts/mlrbench-agent-claude.json"


def manuscripts(task_dir):
    """{writer model: text}. One run's evidence, several writers."""
    out = {}
    for p in sorted((task_dir / "results").glob("paper_*.md")):
        out[p.stem.replace("paper_", "")] = p.read_text(encoding="utf-8")
    return out


def run_task(task_dir, contract):
    rows = []
    for writer, text in manuscripts(task_dir).items():
        uni = phi.extract(task_dir, contract, text=text)
        st = phi.strata(uni["slots"])
        rows.append({
            "task": task_dir.name,
            "writer": writer,
            "artifact_keys": uni["artifact_keys"],
            "N": uni["N"],
            "unresolved": uni["unresolved_claims"],
            "by_cause": uni["unresolved_by_cause"],
            "parse_failure_rate": uni["parse_failure_rate"],
            "not_recorded_rate": uni["not_recorded_rate"],
            "N_by_layer": {k: len(v) for k, v in st.items() if v},
            "universe_digest": phi.universe_digest(uni)[:16],
            "evaluable": uni["N"] >= 10,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--task")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="arms/mlrbench_agent_phi.json")
    a = ap.parse_args()

    root = pathlib.Path(a.repo) / "agent_results/experiments_and_writeups" / BACKEND
    contract = json.loads(CONTRACT.read_text())
    tasks = sorted(p for p in root.iterdir() if p.is_dir())
    if a.task:
        tasks = [t for t in tasks if t.name == a.task]
    elif not a.all:
        tasks = [t for t in tasks if t.name == DEV_TASK]

    rows = []
    hdr = (f"{'task':24s} {'writer':26s} {'keys':>6} {'N':>5} {'unres':>6} "
           f"{'ours':>7} {'theirs':>7} {'eval':>5}")
    print(hdr); print("-" * len(hdr))
    for t in tasks:
        for r in run_task(t, contract):
            rows.append(r)
            print(f"{r['task']:24s} {r['writer'][:26]:26s} {r['artifact_keys']:>6} "
                  f"{r['N']:>5} {r['unresolved']:>6} {r['parse_failure_rate']:>6.1%} "
                  f"{r['not_recorded_rate']:>6.1%} {'YES' if r['evaluable'] else 'NO':>5}")
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))

    if len(rows) > 1:
        Ns = sorted(r["N"] for r in rows)
        ev = sum(1 for r in rows if r["evaluable"])
        print(f"\nevaluable (N>=10): {ev}/{len(rows)}   N median={statistics.median(Ns)}  "
              f"range={Ns[0]}-{Ns[-1]}")
        held = [r for r in rows if r["task"] != DEV_TASK]
        if held:
            print(f"held-out only    : {sum(1 for r in held if r['evaluable'])}/{len(held)}")
    return 0


if __name__ == "__main__":
    main()
