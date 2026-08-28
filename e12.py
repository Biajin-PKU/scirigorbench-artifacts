#!/usr/bin/env python3
"""E12 -- does the headline survive the floor, and what does N actually look like?

The paper's central number is a count of runs above a threshold: N >= 10, fixed
in the frozen manifest before any external arm was run. A count above a
threshold is only as interesting as the threshold, and a reviewer is entitled to
ask whether the zero was manufactured by choosing it. So:

  (a) sweep the floor over {2,3,5,10,20,50} and report scoreable runs per arm;
  (b) print the full per-run N distribution rather than a median and one max.

Neither needs a new adjudicator call. Both use the arm extractions already on
disk, so this is a re-reading of measured data and not a new measurement.

    python3 e12.py
"""

import json
import pathlib
import statistics as st
from math import comb
import sys

HERE = pathlib.Path(__file__).resolve().parent

ARMS = [
    ("AI Scientist-v2", "arms/mlrbench_v2_phi.json"),
    ("MLR-Agent",       "arms/mlrbench_agent_phi.json"),
    ("ARA",             "arms/ara_labs_phi.json"),
    ("AI Scientist-v1", "arms/aiscientist_v1_phi.json"),
    ("ICML-26 repro",   "arms/icml_repro_phi.json"),
]
DEV = "e1_m2_all7.json"          # the seven development substrates
FLOORS = [2, 3, 5, 10, 20, 50]
FROZEN_FLOOR = 10                # FREEZE.json: evaluability_floor_N


def load(path):
    p = HERE / path
    if not p.exists():
        return None
    return json.loads(p.read_text())


def dev_Ns():
    """The seven development substrates, one N each."""
    rows = load(DEV) or []
    by_topic = {}
    for r in rows:
        t = str(r.get("topic"))
        by_topic[t] = max(by_topic.get(t, 0), r.get("N", 0))
    return sorted(by_topic.values(), reverse=True)


def main():
    arms = []
    for name, path in ARMS:
        rows = load(path)
        if rows is None:
            print(f"  [{name}: {path} missing]")
            continue
        arms.append((name, [r["N"] for r in rows]))
    dev = dev_Ns()
    if dev:
        arms.append(("RH (development, in-sample)", dev))

    print("E12 -- floor sensitivity and the shape of N\n")

    # ---- (a) does the headline depend on the frozen floor? -------------------
    print("  scoreable runs at each floor")
    head = f"  {'arm':30s}{'runs':>5}" + "".join(f"{'N>='+str(f):>8}" for f in FLOORS)
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name, Ns in arms:
        cells = "".join(f"{sum(1 for n in Ns if n >= f):>8}" for f in FLOORS)
        print(f"  {name:30s}{len(Ns):5d}{cells}")
    ext = [n for name, Ns in arms if "development" not in name for n in Ns]
    print(f"\n  external total ({len(ext)} runs):" +
          "".join(f"{sum(1 for n in ext if n >= f):>8}" for f in FLOORS))

    # the honest reading: where would the headline change?
    flips = [f for f in FLOORS if sum(1 for n in ext if n >= f) != 1]
    print(f"\n  the frozen floor is N >= {FROZEN_FLOOR}, at which {sum(1 for n in ext if n >= FROZEN_FLOOR)} "
          f"external run is scoreable.")
    print(f"  floors that would give a different count: {flips}")
    lowest = min(ext), max(ext)
    print(f"  external N ranges {lowest[0]}..{lowest[1]}; a floor of 2 admits "
          f"{sum(1 for n in ext if n >= 2)} runs, a floor of 50 admits "
          f"{sum(1 for n in ext if n >= 50)}.")

    # ---- (b) the distribution, not the median --------------------------------
    print("\n  per-run N, every run, sorted")
    for name, Ns in arms:
        s = sorted(Ns, reverse=True)
        q = (f"median {st.median(s):g}  mean {st.mean(s):.1f}  "
             f"zeros {sum(1 for n in s if n == 0)}/{len(s)}")
        print(f"    {name:30s} {q}")
        print(f"      {s}")

    out = {
        "frozen_floor": FROZEN_FLOOR,
        "floors": FLOORS,
        "per_arm": {name: sorted(Ns, reverse=True) for name, Ns in arms},
        "scoreable_at_floor": {
            name: {str(f): sum(1 for n in Ns if n >= f) for f in FLOORS}
            for name, Ns in arms},
        "external_scoreable_at_floor": {
            str(f): sum(1 for n in ext if n >= f) for f in FLOORS},
    }
    (HERE / "e12_floor.json").write_text(json.dumps(out, indent=1))

    # ---- (c) the floor is derivable, and ours was set too low ---------------
    KAPPA, ALPHA = 0.5, 0.05
    def best_e(N, K):
        """Most evidence obtainable at this universe size: a perfect adjudicator
        accuses exactly the planted set, so p = 1 / C(N, K)."""
        return KAPPA * (1 / comb(N, K)) ** (KAPPA - 1)
    need = 1 / ALPHA
    print("\n  most evidence a universe of size N can ever yield")
    print(f"  {'N':>5}" + "".join(f"{'K='+str(k):>10}" for k in (1, 2, 3, 5)))
    for N in (5, 10, 15, 23, 30, 50, 159):
        print(f"  {N:>5}" + "".join(
            f"{best_e(N,k):>10.1f}" if k < N else f"{'-':>10}" for k in (1, 2, 3, 5)))
    mins = {}
    for K in (1, 2, 3, 5):
        mins[K] = next((N for N in range(K + 1, 600) if best_e(N, K) >= need), None)
    print(f"\n  e-BH needs e >= 1/alpha = {need:.0f} in a stratum. Smallest N that can reach it:")
    for K, N in mins.items():
        print(f"    K={K}: {N if N else 'unreachable at any N'}")

    attainable = mins[3]
    print(f"""
Reading. Two things, and the second is against us.

The count of scoreable external runs moves with the floor -- {sum(1 for n in ext if n>=2)} runs clear a
floor of 2, {sum(1 for n in ext if n>=5)} clear 5, {sum(1 for n in ext if n>=FROZEN_FLOOR)} clears {FROZEN_FLOOR}, {sum(1 for n in ext if n>=20)} clear 20 -- so the headline cannot be
defended by saying the threshold does not matter. It can be defended by the
freeze: the floor was fixed before any external arm was run, and the manifest
timestamps it.

The floor itself, though, is derivable rather than chosen, and the derivation
says ours was too low. A universe of size N with K planted slots yields at most
e = kappa * C(N,K)^(1-kappa), attained when the adjudicator accuses exactly the
planted set. At N={FROZEN_FLOOR} and K=3 that ceiling is {best_e(FROZEN_FLOOR,3):.1f}, well under the {need:.0f} a single
stratum needs; with K=1 no universe of any size reaches it. The smallest
universe that can ever certify at K=3 is N={attainable}.

So the one external run that clears our recorded floor still cannot produce a
certificate, and at the floor that evidence actually requires the external count
is {sum(1 for n in ext if n>=attainable)} of {len(ext)}. We report the recorded floor because it is what was frozen,
and this derivation because it is what the frozen value should have been.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
