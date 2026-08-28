#!/usr/bin/env python3
"""E8 -- the preregistered cross-benchmark check, evaluated by its own rules.

The registration is frozen at
`docs/innovation-kb/evidence/e8-preregistration-2026-08-25.md`. Its point is
that a prediction written after the measurement explains any outcome, so the
endpoint, the decision rule and the VOID conditions were all fixed first. This
script does nothing but apply them.

    prediction (section 1)  AI Scientist v2's process fidelity should be low
    unit (section 2.1)      one run; evaluable iff N_run >= 10
    endpoint (section 2.2)  H0: p >= 0.5, one-sided exact binomial, alpha=0.05
    VOID (section 3)        five conditions, any one of which voids the test

Reporting a VOID as VOID is the whole discipline. A voided test is not weak
support and it is not a contradiction, and the registration says so in advance
precisely so that this moment cannot be argued either way after the fact.

    python3 e8.py
"""

import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

# Frozen in the registration; repeated here so the rule and its application sit
# in one place, and so a diff shows if either moves.
EVALUABILITY_FLOOR = 10          # section 2.1
NON_EVALUABLE_VOID = 0.30        # section 3, condition 2
PARSE_FAILURE_VOID_RATIO = 2.0   # section 3, condition 1
ALPHA = 0.05                     # section 2.2
H0_P = 0.5                       # section 2.2, the weaker of MLR-Bench's two sentences

# The development arm's parse failure rate at the same specification, which
# condition 1 measures the external arm against.
RH_PARSE_FAILURE = 0.573


def load(name):
    p = HERE / "arms" / name
    if not p.exists():
        sys.exit(f"missing {p}; run the arm adapters first")
    return json.loads(p.read_text())


def binom_sf(x, n, p):
    """P(X >= x) for Binomial(n, p), exact."""
    return sum(math.comb(n, k) * p**k * (1 - p)**(n - k) for k in range(x, n + 1))


def main():
    v2 = load("mlrbench_v2_phi.json")
    n_total = len(v2)
    evaluable = [r for r in v2 if r["N"] >= EVALUABILITY_FLOOR]
    non_eval_frac = 1 - len(evaluable) / n_total
    parse_fail = [r["parse_failure_rate"] for r in v2]
    parse_median = sorted(parse_fail)[len(parse_fail) // 2]

    print("E8 -- preregistered cross-benchmark check, AI Scientist v2")
    print(f"  runs                    : {n_total}")
    print(f"  evaluable (N >= {EVALUABILITY_FLOOR})     : {len(evaluable)}")
    print(f"  non-evaluable fraction  : {non_eval_frac:.0%}")
    print(f"  parse failure (median)  : {parse_median:.1%}"
          f"   [dev arm {RH_PARSE_FAILURE:.1%}, void above "
          f"{PARSE_FAILURE_VOID_RATIO * RH_PARSE_FAILURE:.1%}]")

    voids = []
    if parse_median > PARSE_FAILURE_VOID_RATIO * RH_PARSE_FAILURE:
        voids.append("1: the instrument cannot read this arm "
                     f"({parse_median:.1%} > {PARSE_FAILURE_VOID_RATIO}x dev arm)")
    if non_eval_frac > NON_EVALUABLE_VOID:
        voids.append(f"2: the denominator collapsed "
                     f"({non_eval_frac:.0%} non-evaluable > {NON_EVALUABLE_VOID:.0%})")

    print()
    if voids:
        print("VERDICT: VOID")
        for v in voids:
            print(f"  condition {v}")
        print()
        print("  Reported as VOID, per section 3: NOT as CONTRADICTED and NOT as")
        print("  support in either direction. The prediction is neither confirmed")
        print("  nor refuted by this run of the test, because the test could not")
        print("  be run on a denominator this thin.")
        print()
        print("  Note which condition fired. Condition 2 is about OUR reach, not")
        print("  about the arm: it says we could not assemble enough scoreable")
        print("  runs, and section 4.8.7 forbids reading that as a finding about")
        print("  the agent's recording discipline.")
    else:
        X = sum(1 for r in evaluable if r.get("hit"))
        n = len(evaluable)
        p_val = binom_sf(X, n, H0_P) if n else 1.0
        print(f"  X/n = {X}/{n}   one-sided exact binomial p = {p_val:.4g}")
        if n and X / n >= H0_P:
            print("VERDICT: CORROBORATED")
        elif p_val < ALPHA:
            print("VERDICT: CONTRADICTED -- publication blocked until resolved (section 2.3)")
        else:
            print("VERDICT: INCONCLUSIVE")

    out = {
        "arm": "ai-scientist-v2",
        "runs": n_total,
        "evaluable": len(evaluable),
        "non_evaluable_fraction": round(non_eval_frac, 4),
        "parse_failure_median": parse_median,
        "void_conditions_fired": voids,
        "verdict": "VOID" if voids else None,
    }
    (HERE / "e8_verdict.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
