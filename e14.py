#!/usr/bin/env python3
"""E14 -- what the design could have detected, and what it cost.

Two things the paper concedes without quantifying.

  (a) POWER. The limitations say "at this power we could not resolve a
      threshold if one existed". That is an admission, not a bound. Here we
      compute the minimum detection rate the design can distinguish from
      chance at the replicate counts actually used, so the admission becomes a
      number a reader can check the design against.

  (b) COST. The construction's premise is that adjudication is the scarce
      resource. That claim is never priced. We count the adjudicator calls the
      design implies, time the deterministic stages, and set both against what
      the arms spent producing the runs we audit.

Honest scope: our own adjudicator's token spend was NOT recorded at the time,
so it is reported as absent rather than reconstructed. Everything else here is
either a design fact, a fresh measurement, or the arms' own ledger.

    python3 e14.py
"""

import json
import pathlib
import statistics as st
import sys
import time
from math import comb

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

KAPPA, ALPHA = 0.5, 0.05
NEED = 1 / ALPHA


def hyper_sf(x, N, K, A):
    if A == 0 or K == 0 or K >= N:
        return 1.0
    tot = comb(N, A)
    return sum(comb(K, i) * comb(N - K, A - i)
               for i in range(x, min(K, A) + 1)) / tot


def e_of(N, K, A, X):
    p = max(hyper_sf(X, N, K, A), 1e-300)
    return KAPPA * p ** (KAPPA - 1)


def expected_product(N, K, q, f, reps):
    """Evidence after `reps` replicates at detection rate q, false rate f,
    taking the rounded expected counts. Deterministic stand-in for the median."""
    X = round(q * K)
    A = X + round(f * (N - K))
    return e_of(N, K, A, X) ** reps


def min_detectable(N, K, reps, f=0.0):
    for q100 in range(0, 101):
        q = q100 / 100
        if expected_product(N, K, q, f, reps) >= NEED:
            return q
    return None


def main():
    print("E14 -- power and cost\n")

    # ---------------------------------------------------------------- power
    N = 159                     # development substrate universe
    print("  (a) minimum detection rate the design can separate from chance")
    print(f"      universe N={N}, target e >= 1/alpha = {NEED:.0f}\n")
    hdr = f"      {'K':>3}" + "".join(f"{str(r)+' reps':>10}" for r in (1, 3, 5, 10, 20))
    print(hdr); print("      " + "-" * (len(hdr) - 6))
    rows = {}
    for K in (1, 2, 3, 5, 10):
        cells = ""
        for reps in (1, 3, 5, 10, 20):
            q = min_detectable(N, K, reps)
            cells += f"{(f'{q:.0%}' if q is not None else 'none'):>10}"
            rows[(K, reps)] = q
        print(f"      {K:>3}{cells}")

    used_K, used_reps = 3, 5
    q_used = rows[(used_K, used_reps)]
    print(f"\n      the sweep in this paper used K={used_K}, {used_reps} replicates per level.")
    print(f"      minimum detectable detection rate there: "
          f"{f'{q_used:.0%}' if q_used is not None else 'no rate is detectable'}")
    q10 = rows[(used_K, 10)]
    print(f"      at 10 replicates it would be "
          f"{f'{q10:.0%}' if q10 is not None else 'still none'}")

    # ------------------------------------------------------------- cost
    print("\n  (b) cost")
    import json as _json
    _ext = sum(len(_json.loads((HERE / f).read_text()))
               for f in ("arms/mlrbench_v2_phi.json", "arms/mlrbench_agent_phi.json",
                         "arms/ara_labs_phi.json", "arms/aiscientist_v1_phi.json",
                         "arms/icml_repro_phi.json") if (HERE / f).exists())
    def _n(f, key=None):
        """Rows in an experiment output, so the count tracks the file."""
        p = HERE / f
        if not p.exists():
            return 0
        d = _json.loads(p.read_text())
        d = d if isinstance(d, list) else d.get(key or "rows", [])
        return len(d)

    # Every model call that produced a certificate line. Deliberately NOT the
    # paper's total model spend: e13 (extractor ceiling), e17 and e18 (writers)
    # call a model but adjudicate nothing, and folding them in here would price
    # the audit using calls the audit does not make.
    calls = {"E1 M2 sweep": _n("e1_m2_all7.json"),
             "E1 M1 sweep": _n("e1_m1.json"),
             "E1 M1 concealment": 34,
             "E1 M1 re-measurement": _n("e1_m1_remeasured_k3.json")
                                    + _n("e1_m1_remeasured_k0.json"),
             "E2 ordering recovery": _n("e2.json"),
             "E15 fixed package": _n("e15_variance.json"),
             "external arms (one adjudication each)": _ext}
    total = sum(calls.values())
    for k, v in calls.items():
        print(f"      {k:42s}{v:5d} adjudications")
    print(f"      {'total':42s}{total:5d}")

    t0 = time.time()
    try:
        import phi
        from e1 import topics_with_substrate
        run = dict(topics_with_substrate()).get("84")
        contract = json.loads((HERE / "contracts/substrate-84.json").read_text())
        t1 = time.time()
        idx = phi.artifact_index(run, None, contract)
        t2 = time.time()
        res = phi.extract(run, contract)
        t3 = time.time()
        print(f"\n      deterministic stages, measured now on the development substrate:")
        print(f"        artifact index ({len(idx)} keys) : {t2-t1:6.2f} s")
        print(f"        claim extraction (N={res['N']})   : {t3-t2:6.2f} s")
        det = t3 - t1
    except Exception as e:
        print(f"      [deterministic timing unavailable: {type(e).__name__}]")
        det = None

    trackers = sorted(pathlib.Path("/tmp/mlrb/r/ai_scientist_v2_papers/o4-mini").glob(
        "*/token_tracker.json")) if pathlib.Path("/tmp/mlrb/r").exists() else []
    if trackers:
        costs, toks = [], []
        for f in trackers:
            for model, blob in json.loads(f.read_text()).items():
                if "cost (USD)" in blob:
                    costs.append(blob["cost (USD)"])
                    t = blob.get("tokens", {})
                    toks.append(sum(v for k, v in t.items() if k != "cached"))
        print(f"\n      what the arm spent PRODUCING each run it we audit "
              f"(its own ledger, {len(costs)} runs):")
        print(f"        tokens  median {st.median(toks):,.0f}")
        print(f"        USD     median {st.median(costs):.2f}   total {sum(costs):.2f}")
        prod = st.median(costs)
    else:
        prod = None
        print("\n      [arm ledgers not mounted]")

    out = {"power": {f"K{K}_reps{r}": rows[(K, r)] for K, r in rows},
           "design_K": used_K, "design_reps": used_reps,
           "min_detectable_at_design": q_used,
           "adjudications": calls, "adjudications_total": total,
           "deterministic_seconds": det,
           "arm_production_usd_median": prod,
           "arm_production_tokens_median": (st.median(toks) if trackers and toks else None),
           "our_adjudicator_tokens": None}
    (HERE / "e14_power_cost.json").write_text(json.dumps(out, indent=1))

    print(f"""
Reading. The concealment sweep used K={used_K} and {used_reps} replicates, where the design can
only separate a detector from chance once it catches {f'{q_used:.0%}' if q_used is not None else 'an unattainable share'} of what was
planted. Reporting that M1 does not certify at any concealment level is
therefore a statement about a coarse instrument -- but not one replicates fix.
With K={used_K} the hit count is one of 0,1,2,3, so the detection rate is quantised at
thirds and the bar stays at {f'{q_used:.0%}' if q_used is not None else 'none'} however many replicates are run. What moves it is
K: at K=10 and {used_reps} replicates the bar is {f'{rows[(10,used_reps)]:.0%}' if rows[(10,used_reps)] is not None else 'none'}. The resolution of the concealment
sweep is set by the planting budget, not by repetition, and that is a design
fact we should have derived before running it rather than after.

On cost, the whole audit is {total} adjudications; the deterministic stages that
surround them run in {f'{det:.1f} s' if det else 'an unmeasured time'} on the development substrate. Set against
{f'a median of ${prod:.2f} per run' if prod else 'the arms  own ledgers'} to produce what we audit, the claim that adjudication is
the scarce resource holds -- but our own adjudicator's token spend was never
recorded, so the comparison is one-sided and is reported as such.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
