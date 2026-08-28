#!/usr/bin/env python3
"""E4 -- ranking stability under a perturbation that should not change the answer.

MLR-Bench generated three manuscripts from each run: the experiment, the code
and the artifacts are identical, and only the model writing the paper differs.
That is a rare thing to have. A score of the RESEARCH should be nearly invariant
under it, because the research is the same object; anything that moves is
measuring the write-up.

Two methods are compared on the same perturbation:

    MLR-Judge   rubric scores (Clarity, Novelty, Soundness, Significance,
                Overall) from human_eval/llm_judge.csv
    ours        N, the number of auditable claims

The comparison is not "who is right". The rubric is trying to score the
research, so movement under a writer swap is instability. N is explicitly a
property of the write-up, so movement is the measurement working. The point of
putting them side by side is that only one of the two reports enough to notice
it moved.

    python3 e4.py --repo /tmp/mlrb/r
"""

import argparse
import csv
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent

AXES = ["Clarity", "Novelty", "Soundness", "Significance", "Overall"]
WRITERS = {"claude": "claude-3-7-sonnet-20250219",
           "gemini": "gemini-2.5-pro-preview",
           "o4-mini": "o4-mini-2025-04-16"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    a = ap.parse_args()

    judge = {}
    with open(pathlib.Path(a.repo) / "human_eval/llm_judge.csv") as fh:
        for r in csv.DictReader(fh):
            judge[(r["workshop_name"], r["model_name"])] = {
                ax: float(r[ax]) for ax in AXES if r.get(ax)}

    ours = {}
    for r in json.loads((HERE / "arms/mlrbench_agent_phi.json").read_text()):
        short = next((k for k, v in WRITERS.items() if v == r["writer"]), None)
        if short:
            ours[(r["task"], short)] = r["N"]

    tasks = sorted({t for t, _ in judge})
    print("E4 -- same run, three writers. What moves?\n")
    print(f"{'task':24s} {'Overall (c/g/o)':>18} {'spread':>7} "
          f"{'N (c/g/o)':>14} {'best differs':>13}")
    print("-" * 82)

    spreads, flips, pairs = [], 0, 0
    for t in tasks:
        ov = [judge.get((t, w), {}).get("Overall") for w in WRITERS]
        ns = [ours.get((t, w)) for w in WRITERS]
        if None in ov or None in ns:
            continue
        spread = max(ov) - min(ov)
        spreads.append(spread)
        best_j = max(range(3), key=lambda i: ov[i])
        best_n = max(range(3), key=lambda i: ns[i])
        differs = best_j != best_n
        flips += differs
        pairs += 1
        print(f"{t:24s} {'/'.join(f'{x:.1f}' for x in ov):>18} {spread:>7.1f} "
              f"{'/'.join(str(x) for x in ns):>14} {'YES' if differs else '':>13}")

    print()
    print(f"  rubric Overall spread across writers, same research:")
    print(f"    median {statistics.median(spreads):.1f}   max {max(spreads):.1f}"
          f"   on a scale where the whole corpus spans "
          f"{min(v['Overall'] for v in judge.values()):.1f}"
          f"-{max(v['Overall'] for v in judge.values()):.1f}")
    print(f"  best writer by rubric differs from best by N: {flips}/{pairs} tasks")

    per_axis = {}
    for ax in AXES:
        sp = []
        for t in tasks:
            vs = [judge.get((t, w), {}).get(ax) for w in WRITERS]
            if None not in vs:
                sp.append(max(vs) - min(vs))
        if sp:
            per_axis[ax] = statistics.median(sp)
    print("\n  median spread by axis (research identical in every case):")
    for ax, v in sorted(per_axis.items(), key=lambda kv: -kv[1]):
        print(f"    {ax:14s} {v:>5.1f}  {'#' * int(v * 6)}")

    print(f"""
Reading. The rubric moves by a median of {statistics.median(spreads):.1f} points
of Overall when nothing about the research changed, and {max(spreads):.1f} at the
extreme. Soundness -- the axis that most plausibly refers to the experiment
rather than the prose -- moves by {per_axis.get('Soundness', 0):.1f}. Whatever
that number is measuring, a substantial part of it is the writer.

That is not a claim that the rubric is wrong, and we cannot make one: it has no
error bar, so there is no quantity to compare a movement against. That is the
whole point. A score that moves under a null perturbation and reports no
uncertainty cannot tell its reader, or its authors, that it moved.

Our own column is not exonerated by this. N moves too, and on {flips} of {pairs}
tasks it disagrees with the rubric about which write-up is best. The difference
is that N is declared as a property of the write-up and is reported beside the
denominator it was computed from, so a reader can see the movement and price it.""")

    (HERE / "e4_stability.json").write_text(json.dumps(
        {"spread_median": statistics.median(spreads), "spread_max": max(spreads),
         "per_axis_median_spread": per_axis, "best_differs": flips,
         "tasks": pairs}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
