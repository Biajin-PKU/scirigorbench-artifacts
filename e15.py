#!/usr/bin/env python3
"""E15 -- how much does the adjudicator move when nothing else does?

Every external certificate in this paper is one call to one adjudicator on one
sealed package. The paper asks the field to accept error rates derived from a
stochastic LLM judge, and has not shown what that judge does when asked the same
question twice.

So: fix ONE planted package -- same substrate, same seed, same planted set, same
concealment -- and adjudicate it k times independently. Everything that moves is
the adjudicator.

This is also the empirical face of Proposition A. The proposition needs the
adjudicator to be stateless across replicates; each call here is a fresh
context, so the run doubles as a check that fresh contexts really do behave
independently rather than merely being labelled that way.

    python3 e15.py --k 3 --reps 6
"""

import argparse
import json
import os
import pathlib
import statistics as st
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phi                                                     # noqa: E402
from e1 import certify, topics_with_substrate                  # noqa: E402
from m1 import ask, map_to_slots                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="contracts/substrate-84.json")
    ap.add_argument("--topic", default="84")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--delta", type=float, default=0.30)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--out", default="e15_variance.json")
    a = ap.parse_args()

    from research_harness.env_bootstrap import ensure_default_env_loaded
    ensure_default_env_loaded()
    base = os.environ["OPENAI_BASE_URL"].rstrip("/")
    key = os.environ["OPENAI_API_KEY"]
    model = os.environ.get("LLM_ROUTE_HEAVY", "openai:gpt-5.6-sol").split(":", 1)[-1]

    contract = json.loads((HERE / a.contract).read_text())
    sub = dict(topics_with_substrate())[a.topic]
    uni = phi.extract(sub, contract)

    # ONE package. Built once, outside the loop, so that nothing but the
    # adjudicator differs between rows.
    planted_text, ledger = phi.plant(sub, a.k, a.delta, str(a.seed), contract)
    planted = {p["slot"] for p in ledger["planted"]}

    print(f"E15 -- adjudicator variance on a single sealed package\n")
    print(f"  substrate {a.topic}   N={uni['N']}   K={len(planted)}   "
          f"delta={a.delta}   model={model}")
    print(f"  the package is built once; every row below is the same bytes.\n")
    hdr = f"  {'rep':>4}{'raw':>6}{'A':>5}{'X':>4}{'unmapped':>10}{'p':>10}{'e':>10}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    rows = []
    for rep in range(a.reps):
        try:
            resp = ask(planted_text, model, base, key)
        except Exception as exc:
            print(f"  {rep:>4}  [{type(exc).__name__}] {str(exc)[:40]}")
            continue
        accs = resp.get("accusations", [])
        hit, unmapped = map_to_slots(accs, uni["slots"], planted_text)
        r = certify(list(uni["slots"]), planted, hit, "M1")
        r.update(rep=rep, raw=len(accs), unmapped=len(unmapped),
                 hits=sorted(hit & planted))
        rows.append(r)
        print(f"  {rep:>4}{len(accs):>6}{r['A']:>5}{r['X']:>4}{len(unmapped):>10}"
              f"{r['p']:>10.3g}{r['e']:>10.3g}")
        pathlib.Path(HERE / a.out).write_text(json.dumps(rows, indent=1))

    if len(rows) < 2:
        print("\n  too few completed calls to report a spread")
        return 1

    A = [r["A"] for r in rows]; X = [r["X"] for r in rows]; E = [r["e"] for r in rows]
    raw = [r["raw"] for r in rows]
    certified = sum(1 for e in E if e >= 20)
    # which planted slots were caught, and did the same ones get caught each time?
    sets = [frozenset(r["hits"]) for r in rows]
    agree = len(set(sets)) == 1

    print(f"""
  over {len(rows)} adjudications of identical bytes:
    raw accusations   {min(raw)}..{max(raw)}   median {st.median(raw):g}
    mapped A          {min(A)}..{max(A)}   median {st.median(A):g}
    hits X            {min(X)}..{max(X)}   median {st.median(X):g}
    e                 {min(E):.3g}..{max(E):.3g}   median {st.median(E):.3g}
    certifies (e>=20) {certified}/{len(rows)}
    same planted slots caught every time: {'yes' if agree else 'no'}""")

    (HERE / a.out).write_text(json.dumps(
        {"substrate": a.topic, "N": uni["N"], "K": len(planted), "delta": a.delta,
         "model": model, "reps": len(rows), "rows": rows,
         "spread": {"A": [min(A), max(A)], "X": [min(X), max(X)],
                    "e": [min(E), max(E)]},
         "certified": certified, "same_hits_every_time": agree}, indent=1))

    print(f"""
Reading. Nothing in the package changed between these rows, so the range in e
is the adjudicator alone. A certificate reported from a single call inherits
that range, and this paper has reported single-call certificates for every
external arm. The honest fix is to report the spread, or to fix the decision by
majority over k calls and say k; asserting a single e-value as though it were a
property of the arm is the thing this experiment shows we cannot do.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
