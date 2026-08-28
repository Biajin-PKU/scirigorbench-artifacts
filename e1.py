#!/usr/bin/env python3
"""E1 -- the killer experiment: what do the artifacts buy you?

Two auditors face the SAME frozen package and the same slot universe.  One can
see the run's artifacts; the other sees only the finished manuscript.  Both feed
the same certificate.  The question E1 answers is not "who finds more" -- it is
"who can certify anything at all".

    M2  process adjudicator   sees manuscript + frozen artifacts   deterministic
    M1  manuscript reviewer   sees the manuscript only             LLM

Planting breaks the claim/artifact link and rewrites EVERY printed copy of the
claim coherently, so the manuscript stays internally consistent.  That is not us
being unfair to M1: it is what fabrication actually looks like.  A generator
that invents a number writes it the same way everywhere; only the artifacts
disagree.  MLR-Bench (2505.19955, Appendix C) is the external evidence that
judges reading finished papers are put in exactly this position.

The certificate is adopted from the A paper unchanged -- see
reviewer-cert/data/cert_validation.py.  B contributes the universe, not the test.

Usage:
    python3 e1.py --contract contracts/substrate-84.json          # M2 only
    python3 e1.py --contract contracts/substrate-84.json --k 1 2 3 5 --reps 4
"""

import argparse
import json
import math
import pathlib
from fractions import Fraction

import phi

ARENA = pathlib.Path("substrates")
KAPPA = 0.5


# --- the certificate, verbatim from reviewer-cert/data/cert_validation.py -----
# Exact hypergeometric conditional p-value, then Vovk-Wang calibration into an
# e-value.  Not reimplemented, not retuned: the A paper proves this machinery
# and B adopts it. If this block ever needs to change, B is re-deriving the
# engine, which the plan flags as a sign B does not stand on its own.

def hyper_sf(x, N, K, A):
    if A == 0 or K == 0 or K >= N:
        return Fraction(1)
    tot = Fraction(math.comb(N, A))
    return sum(Fraction(math.comb(K, i) * math.comb(N - K, A - i))
               for i in range(x, min(K, A) + 1)) / tot


def calibrate(p):
    """Vovk-Wang: e = κ p^{κ-1}. At p=1 (including K=0) this is κ, not 1."""
    return KAPPA * max(float(p), 1e-12) ** (KAPPA - 1)
# -----------------------------------------------------------------------------


def m2_accuse(universe, planted_text, run_dir, contract):
    """Process adjudicator: every slot whose claim no longer matches its artifact.

    Deterministic, no model.  A slot is accused when the value printed in the
    manuscript differs from the artifact value at the slot's own printed
    precision -- comparing at the coarsest precision the manuscript used, since
    a claim cannot be wrong about digits it never printed.
    """
    after = phi.extract(run_dir, contract, text=planted_text)["slots"]
    accused = set()
    for name, slot in after.items():
        dp = min(p["dp"] for p in slot["positions"])
        claimed = round(float(slot["positions"][0]["literal"]), dp)
        actual = round(float(slot["artifact_value"]), dp)
        if claimed != actual:
            accused.add(name)
    return accused


def certify(universe_names, planted_names, accused, label):
    """One certificate line: (N, K, A, X) -> exact p -> e-value."""
    N = len(universe_names)
    K = len(planted_names)
    A = len(accused & set(universe_names))
    X = len(accused & set(planted_names))
    p = float(hyper_sf(X, N, K, min(A, N)))
    return {"auditor": label, "N": N, "K": K, "A": A, "X": X,
            "p": p, "e": calibrate(p)}


def topics_with_substrate():
    out = []
    for ep in sorted(ARENA.glob("substrate-*")):
        rounds = sorted(ep.glob("blind_reviews/round-*"),
                        key=lambda p: int(p.name.split("-")[1]))
        if rounds and (rounds[-1] / "submission").exists():
            out.append((ep.name.split("-")[-1], rounds[-1] / "submission"))
    return out


def contract_for(topic, default):
    """Per-run contract when one exists.  D3 face B is declared per arm, and a
    declaration written for one run's file layout does not fit another's."""
    c = pathlib.Path(f"contracts/arena-substrate-topic{topic}.json")
    return json.loads(c.read_text()) if c.exists() else default


def run(default_contract, ks, reps, seed, out=None):
    rows, skipped = [], []
    for topic, sub in topics_with_substrate():
        contract = contract_for(topic, default_contract)
        base = phi.extract(sub, contract)
        if base["N"] < 10:                      # D3 section 3, evaluability
            skipped.append((topic, base["N"], base["unresolved_claims"]))
            continue
        for k in ks:
            for rep in range(reps):
                s = f"{seed}{rep:02d}"
                planted_text, ledger = phi.plant(sub, k, 0.30, s, contract)
                planted = {p["slot"] for p in ledger["planted"]}
                accused = m2_accuse(base, planted_text, sub, contract)
                r = certify(list(base["slots"]), planted, accused, "M2")
                r.update(topic=topic, rep=rep, phi_coverage=round(
                    base["N"] / max(1, base["N"] + base["unresolved_claims"]), 3))
                rows.append(r)
                # Persist per row, not at the end: m1.py lost fifty minutes of
                # a sweep to one dropped connection on its final call because it
                # wrote only once. This loop is cheap but not free, and the
                # failure mode is identical.
                if out:
                    pathlib.Path(out).write_text(json.dumps(rows, indent=1))
    return rows, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out", default="e1_m2.json")
    a = ap.parse_args()

    contract = json.loads(pathlib.Path(a.contract).read_text())
    rows, skipped = run(contract, a.k, a.reps, a.seed, out=a.out)
    pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))

    if skipped:
        print("skipped (N < 10, not evaluable -- reported, not hidden):")
        for t, n, unres in skipped:
            print(f"   topic {t}: N={n:<4} unresolved={unres:<4} "
                  f"Phi coverage={n/max(1,n+unres):.0%}  <- OUR limitation, not the arm's")

    print(f"\n{'topic':>5} {'N':>4} {'K':>3} {'A':>4} {'X':>3} "
          f"{'p':>10} {'e':>8}  {'Phi':>5}")
    for r in rows:
        print(f"{r['topic']:>5} {r['N']:>4} {r['K']:>3} {r['A']:>4} {r['X']:>3} "
              f"{r['p']:>10.3g} {r['e']:>8.3g}  {r['phi_coverage']:>5.0%}")

    # The null arm is the one that can invalidate the instrument: with K=0 the
    # certificate must not fire. Any accusation there is a false accusation.
    print("\nnull check (K=0):")
    for topic, sub in topics_with_substrate():
        c = contract_for(topic, contract)
        base = phi.extract(sub, c)
        if base["N"] < 10:
            continue
        null_text, _ = phi.plant(sub, 0, 0.30, a.seed, c)
        acc = m2_accuse(base, null_text, sub, c)
        flag = "OK" if not acc else f"!! {len(acc)} false accusations"
        print(f"   topic {topic}: N={base['N']}  accused={len(acc)}  {flag}")


if __name__ == "__main__":
    main()
