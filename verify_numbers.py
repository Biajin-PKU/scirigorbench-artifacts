#!/usr/bin/env python3
"""verify_numbers -- recompute every headline number and check it is in the paper.

The manuscript carries a provenance appendix listing where each number came
from. A list is not a check: it did not catch the coverage figure quoted after
it had been superseded, the padding factor reported as two different values, or
the parse-failure rate given three ways. A paper whose subject is unverifiable
numbers cannot ship a provenance table that nothing verifies.

So this recomputes each headline value from the experiment output that produced
it and asserts the value appears in main.tex. It is deliberately dumb: it does
not parse LaTeX, it asks whether the string is present. That catches the failure
mode that actually occurred -- a number updated in one place and not another.

    python3 verify_numbers.py          # exit 1 if anything is missing
"""

import json
import pathlib
import re
import statistics as st
import sys

HERE = pathlib.Path(__file__).resolve().parent
TEX = HERE / "paper" / "main.tex"


def load(name):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v):
    """Render a number the way the manuscript would."""
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.1f}" if abs(v) >= 1 else f"{v:.2f}"
    return str(v)


def main():
    tex = TEX.read_text()
    checks = []

    def want(label, value, *alts):
        """The value must appear, in one of the renderings a writer might pick."""
        cands = {fmt(value), *[fmt(a) for a in alts]}
        if isinstance(value, float):
            cands |= {str(int(round(value))), f"{value:.1f}", f"{value:.2f}"}
        found = any(re.search(r"\\RM\{" + re.escape(c) + r"\}", tex) for c in cands)
        checks.append((label, sorted(cands)[0], found))

    arms = {n: load(f) for n, f in [
        ("AI Scientist-v2", "arms/mlrbench_v2_phi.json"),
        ("MLR-Agent", "arms/mlrbench_agent_phi.json"),
        ("ARA", "arms/ara_labs_phi.json"),
        ("AI Scientist-v1", "arms/aiscientist_v1_phi.json"),
        ("ICML-26 repro", "arms/icml_repro_phi.json")]}
    arms = {k: v for k, v in arms.items() if v}
    allN = [r.get("N", 0) for d in arms.values() for r in d]
    want("external runs, total", len(allN))
    want("external N, maximum", max(allN))
    want("external sources", len(arms))
    for name, d in arms.items():
        Ns = [r.get("N", 0) for r in d]
        want(f"{name}: runs", len(Ns))
        want(f"{name}: median N", st.median(Ns))
        want(f"{name}: max N", max(Ns))

    if (e1 := load("e1_m2_all7.json")):
        want("development substrate N", max(r["N"] for r in e1))
    if (e4 := load("e4_stability.json")):
        want("rubric Overall spread, median", e4["spread_median"])
        want("rubric Overall spread, max", e4["spread_max"])
    if (e7 := load("e7_padding.json")):
        want("padding inflation, max", max(r["inflation"] for r in e7))
        want("padding N, max", max(r["N"] for r in e7))
    if (e15 := load("e15_variance.json")):
        lo, hi = e15["spread"]["e"]
        want("adjudicator e, min", round(lo, 1))
        want("adjudicator e, max", hi)
        want("adjudicator replicates", e15["reps"])
    if (e17 := load("e17_compliance.json")):
        want("compliant bindings, total", sum(r["bound"] for r in e17["rows"]))
        want("compliant bindings, median",
             st.median([r["bound"] for r in e17["rows"]]))
    if (e13 := load("e13_ceiling.json")):
        want("ceiling verified, median", e13["median_verified"])

    # --- staleness: a derived file that aggregates over the arms must have
    # seen all of them. Every failure this script missed was a file left behind
    # when two arms were appended.
    n_runs = len(allN)
    stale = []
    # K=0 rows must carry the calibrator floor kappa, not the historical e=1 special case
    for fname in ("e1_m1_remeasured_k0.json", "e1_m1.json"):
        d = load(fname)
        if not d: continue
        rows = d if isinstance(d, list) else d.get("rows", [])
        for r in rows:
            if r.get("K") == 0 and abs(float(r.get("e", 0)) - 0.5) > 1e-9:
                stale.append(f"{fname} K=0 row has e={r.get('e')}, need kappa=0.5")

    e12 = load("e12_floor.json")
    if e12:
        got = sum(e12.get("scoreable_at_floor", {}).get(k, {}).get("2", 0) * 0 or
                  len(v) for k, v in (e12.get("per_arm") or {}).items())
        ext = sum(len(v) for k, v in (e12.get("per_arm") or {}).items()
                  if "development" not in k)
        if ext != n_runs:
            stale.append(f"e12_floor.json aggregates {ext} external runs, arms hold {n_runs}")
    e14 = load("e14_power_cost.json")
    if e14:
        calls = e14.get("adjudications", {})
        ext_calls = calls.get("external arms (one adjudication each)")
        if ext_calls is not None and ext_calls != n_runs:
            stale.append(f"e14 counts {ext_calls} external adjudications, arms hold {n_runs}")
        tot = e14.get("adjudications_total")
        if tot is not None and tot != sum(calls.values()):
            stale.append(f"e14 total {tot} != sum of its parts {sum(calls.values())}")
    e20 = load("e20_predictors.json")
    if e20:
        seen = e20.get("substrates", 0) + len(e20.get("excluded", []))
        if seen and seen < n_runs:
            stale.append(f"e20 characterises {seen} substrates against {n_runs} external runs")

    # --- arithmetic between reported numbers
    arith = []
    if e14:
        c = e14.get("adjudications", {})
        if c and sum(c.values()) != e14.get("adjudications_total"):
            arith.append("e14 adjudication parts do not sum to its total")

    width = max(len(c[0]) for c in checks)
    print(f"{'quantity':{width}s}  {'value':>10}  source vs manuscript")
    print("-" * (width + 34))
    for label, value, ok in checks:
        print(f"{label:{width}s}  {value:>10}  {'present' if ok else 'MISSING'}")

    if stale:
        print("\nSTALE DERIVED FILES:")
        for m in stale:
            print("  ", m)
    if arith:
        print("\nARITHMETIC:")
        for m in arith:
            print("  ", m)

    missing = [c for c in checks if not c[2]]
    print(f"\n{len(checks) - len(missing)}/{len(checks)} recomputed values appear in the manuscript")
    if missing:
        print("missing:", ", ".join(m[0] for m in missing))
    return 1 if (missing or stale or arith) else 0


if __name__ == "__main__":
    sys.exit(main())
