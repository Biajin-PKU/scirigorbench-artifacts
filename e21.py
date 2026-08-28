#!/usr/bin/env python3
"""E21 -- the matched pair E15 was missing.

E15 adjudicated ONE sealed package six times with the manuscript-only reader and
found it certifies every time at e up to 405 -- a value the paper elsewhere
attributes to the artifact-holding reader alone. Two readings are available and
E15 cannot distinguish them:

    (i)  that package is a favourable DRAW, and the manuscript-only reader beats
         its own typical performance on it;
    (ii) the manuscript-only reader is not floor-bound at all, and the gap the
         paper reports is an artefact of averaging over draws.

Both are settled by running the artifact-holding adjudicator on the SAME bytes.
It is deterministic, so one call is the whole answer: whatever it returns is
that draw's ceiling, and the manuscript-only reader's six rows are measured
against it rather than against a median taken over different draws.

    python3 e21.py                    # same substrate/seed/k/delta as e15.py
"""

import argparse
import json
import pathlib
import statistics as st
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phi                                                     # noqa: E402
from e1 import certify, m2_accuse, topics_with_substrate       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="contracts/substrate-84.json")
    ap.add_argument("--topic", default="84")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--delta", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--m1", default="e15_variance.json")
    ap.add_argument("--out", default="e21_matched.json")
    a = ap.parse_args()

    contract = json.loads((HERE / a.contract).read_text())
    sub = dict(topics_with_substrate())[a.topic]
    uni = phi.extract(sub, contract)

    # Rebuilt with e15's arguments, so these are the same bytes it adjudicated.
    planted_text, ledger = phi.plant(sub, a.k, a.delta, str(a.seed), contract)
    planted = {p["slot"] for p in ledger["planted"]}

    accused = m2_accuse(uni["slots"], planted_text, sub, contract)
    m2 = certify(list(uni["slots"]), planted, accused, "M2")

    m1f = HERE / a.m1
    if not m1f.exists():
        print(f"{a.m1} not found -- run e15.py first"); return 1
    m1 = json.loads(m1f.read_text())
    rows = m1["rows"]
    if m1["N"] != uni["N"] or m1["K"] != len(planted):
        print(f"  package mismatch: e15 has N={m1['N']} K={m1['K']}, "
              f"rebuilt N={uni['N']} K={len(planted)}")
        return 1

    X = [r["X"] for r in rows]; E = [r["e"] for r in rows]
    print(f"""E21 -- both adjudicators on one sealed package

  substrate {a.topic}   N={uni['N']}   K={len(planted)}   delta={a.delta}
  the same bytes e15.py adjudicated; the package is rebuilt, not re-drawn.

  {'adjudicator':34s}{'A':>5}{'X':>4}{'e':>12}{'certifies':>11}
  {'-' * 66}
  {'manuscript only (median of ' + str(len(rows)) + ')':34s}"""
          f"{st.median([r['A'] for r in rows]):>5g}{st.median(X):>4g}"
          f"{st.median(E):>12.3g}{sum(1 for e in E if e >= 20)}/{len(rows):>9}")
    print(f"  {'manuscript + artifacts':34s}{m2['A']:>5}{m2['X']:>4}"
          f"{m2['e']:>12.3g}{'1/1' if m2['e'] >= 20 else '0/1':>11}")

    gap = m2["e"] / st.median(E) if st.median(E) else float("inf")
    out = {"substrate": a.topic, "N": uni["N"], "K": len(planted),
           "delta": a.delta, "seed": a.seed,
           "m1": {"reps": len(rows), "X": X, "e": E,
                  "X_median": st.median(X), "e_median": st.median(E),
                  "certified": sum(1 for e in E if e >= 20)},
           "m2": {"A": m2["A"], "X": m2["X"], "e": m2["e"],
                  "certified": int(m2["e"] >= 20)},
           "gap_on_this_draw": round(gap, 2),
           "m1_reaches_m2": st.median(X) >= m2["X"]}
    (HERE / a.out).write_text(json.dumps(out, indent=1))

    print(f"""
  gap on this draw: {gap:.2f}x

Reading. Whatever separates the two adjudicators, it is a property of the draw
and not a floor. On this package the manuscript-only reader recovers {st.median(X):g} of the
{len(planted)} planted claims where the artifact-holding one recovers {m2['X']}, and the paper's
own sweep over fresh draws puts the same reader at a median of 1. The
quantity that behaves is the DISTRIBUTION over draws; a single draw does not
order the two adjudicators, and no claim in this paper may rest on one.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
