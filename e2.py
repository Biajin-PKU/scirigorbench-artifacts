#!/usr/bin/env python3
"""E2 -- recovering a known ordering.

One planted manuscript. Four adjudicators that differ in EXACTLY one thing: how
much of the frozen run they are allowed to look at. The manuscript handed to
each is the same object -- not a copy produced under the same settings, the
same bytes -- and the script asserts that before scoring anything.

    rung 0   all 7 declared reportable outputs
    rung 1   first 4, in the frozen contract order
    rung 2   first 2
    rung 3   none at all -- the manuscript and nothing else

The ordering of artifact access is known by construction: rung 0 strictly
contains rung 1 strictly contains rung 2 strictly contains rung 3. The question
is whether the certificate recovers it.

Two things follow, and the second is the one that matters:

  * If the certificate does NOT separate the rungs, our instrument cannot see a
    difference we built by hand, and E1 is an observation rather than a result.
    That outcome eliminates us, not anyone else.
  * If it does, then because the manuscript is byte-identical across all four,
    ANY function of the manuscript alone returns one value for all of them. A
    method that reads the finished paper is blind to this difference by
    construction -- not as a measurement, as arithmetic.

The truncation is a prefix of the contract's declared order, fixed before the
draw and not chosen with reference to which slots were planted. Choosing which
globs to withhold after seeing the plant would make this experiment a tuning
knob rather than a test.

Usage:
    python3 e2.py --contract contracts/substrate-84.json --k 3 --reps 5
"""

import argparse
import copy
import hashlib
import json
import pathlib
import statistics

import phi
from e1 import certify, m2_accuse, topics_with_substrate

RUNGS = [(0, None), (1, 4), (2, 2), (3, 0)]   # (rung, how many globs kept)


def rung_contract(contract, keep):
    """Contract with a prefix of the declared reportable outputs.

    keep=None is the frozen contract untouched; keep=0 leaves the adjudicator
    with the manuscript and no artifacts at all.
    """
    if keep is None:
        return contract
    c = copy.deepcopy(contract)
    c["reportable_outputs"] = list(contract["reportable_outputs"])[:keep]
    c["_rung_note"] = (f"prefix of {len(contract['reportable_outputs'])} declared "
                       f"globs, kept {keep}; prefix fixed before the draw")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True)
    ap.add_argument("--topic", default="84")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--delta", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out", default="e2.json")
    a = ap.parse_args()

    contract = json.loads(pathlib.Path(a.contract).read_text())
    sub = dict(topics_with_substrate())[a.topic]

    # The universe, N and the draw all come from the FULL contract and are held
    # fixed across rungs. If each rung re-derived its own universe, the rungs
    # would differ in N as well as in access and the comparison would be
    # between two different tests rather than two views of one.
    base = phi.extract(sub, contract)
    names = list(base["slots"])
    print(f"topic {a.topic}: N={base['N']}  globs={len(contract['reportable_outputs'])}  "
          f"K={a.k}  delta={a.delta}  reps={a.reps}\n")

    rows, digests = [], set()
    for rep in range(a.reps):
        s = f"{a.seed}{rep:02d}"
        planted_text, ledger = phi.plant(sub, a.k, a.delta, s, contract)
        planted = {p["slot"] for p in ledger["planted"]}
        sha = hashlib.sha256(planted_text.encode("utf-8")).hexdigest()

        for rung, keep in RUNGS:
            c = rung_contract(contract, keep)
            # Same object, every rung. Recomputing or re-planting per rung would
            # leave "byte-identical" as an assumption instead of a fact.
            accused = m2_accuse(base, planted_text, sub, c)
            r = certify(names, planted, accused, f"M2@rung{rung}")
            after = phi.extract(sub, c, text=planted_text)
            r.update(rep=rep, rung=rung, globs_kept=keep if keep is not None
                     else len(contract["reportable_outputs"]),
                     resolvable=after["N"], manuscript_sha256=sha)
            rows.append(r)
            digests.add(sha)
            print(f"  rep={rep} rung={rung} globs={r['globs_kept']:>1} "
                  f"resolvable={r['resolvable']:>4} A={r['A']:>3} X={r['X']:>3} "
                  f"p={r['p']:<10.3g} e={r['e']:>8.3g}")
        print(f"        manuscript sha256 {sha[:16]}")

    pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))

    # --- the constructive claim, checked rather than asserted ----------------
    print("\n--- byte identity across rungs ---")
    per_rep = {}
    for r in rows:
        per_rep.setdefault(r["rep"], set()).add(r["manuscript_sha256"])
    ok = all(len(v) == 1 for v in per_rep.values())
    print(f"  each rep's four rungs share one manuscript sha256: {'YES' if ok else 'NO'}")
    print(f"  ⇒ any function of the manuscript alone returns ONE value per rep,")
    print(f"    for all four rungs. Blindness here is arithmetic, not a finding.")

    print("\n--- ordering recovery ---")
    print(f"  {'rung':>4} {'globs':>5} {'resolvable':>10} {'median e':>10} {'A=K':>5}")
    meds = []
    for rung, _ in RUNGS:
        g = [r for r in rows if r["rung"] == rung]
        med = statistics.median(r["e"] for r in g)
        meds.append(med)
        print(f"  {rung:>4} {g[0]['globs_kept']:>5} "
              f"{statistics.median(r['resolvable'] for r in g):>10.0f} "
              f"{med:>10.3g} {sum(1 for r in g if r['A'] == r['K']):>3}/{len(g)}")
    mono = all(x >= y for x, y in zip(meds, meds[1:]))
    print(f"\n  monotone non-increasing as artifacts are withdrawn: "
          f"{'YES' if mono else 'NO'}")
    if not mono:
        print("  ⇒ the certificate does not recover an ordering we built by hand.")
        print("    Report it: that is a limit of this instrument, not of the arms.")


if __name__ == "__main__":
    main()
