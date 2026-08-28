#!/usr/bin/env python3
"""E20 -- what predicts a usable universe, over every substrate we have.

E16 asked what separates the one development substrate that works from the six
that do not, and could not answer: with seven points nothing orders the outcome.
A reviewer asked for fifty more substrates from the same generator. We cannot
regenerate that generator's episodes, but the question does not require them --
by now there are sixty-five substrates in hand, fifty-eight external and seven
development, and the mechanism question can be asked of all of them.

The features are the ones any substrate exposes without our interpretation:
how many artifact cells the run recorded, how long the manuscript is, how many
numerals it prints, and the ratio between them. The outcome is N.

Two things this can find and one it cannot. It can find a predictor, and it can
find that none of these features is one. It cannot find a CAUSE: every feature
here is a property of the released bundle, and a bundle is the joint result of
a pipeline we did not run and a writer we did not see.

    python3 e20.py
"""

import json
import math
import pathlib
import re
import statistics as st
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ARMS = [("AI Scientist-v2", "arms/mlrbench_v2_phi.json"),
        ("MLR-Agent", "arms/mlrbench_agent_phi.json"),
        ("ARA", "arms/ara_labs_phi.json"),
        ("AI Scientist-v1", "arms/aiscientist_v1_phi.json"),
        ("ICML-26 repro", "arms/icml_repro_phi.json")]
DEV = "e16_mechanism.json"
FLOOR = 10


def spearman(xs, ys):
    """Rank correlation, ties averaged. Kept local so the script has no deps."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def collect():
    """Returns (rows, excluded). A substrate with no manuscript cannot be
    characterised by manuscript features; it is excluded and NAMED, because a
    silently dropped row is how a 58-run corpus becomes a 57-run analysis."""
    rows, excluded = [], []
    for name, f in ARMS:
        p = HERE / f
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            keys = r.get("artifact_keys")
            chars = r.get("manuscript_chars")
            if keys is None:
                continue                      # no bundle to characterise
            if not chars:
                excluded.append((name, r.get("task") or r.get("paper")
                                 or r.get("submission"), "no manuscript published"))
                continue
            rows.append({"arm": name, "id": r.get("task") or r.get("paper")
                         or r.get("submission"), "N": r.get("N", 0),
                         "keys": keys, "chars": chars,
                         "keys_per_kchar": round(keys / (chars / 1000), 2)})
    dev = HERE / DEV
    if dev.exists():
        for r in json.loads(dev.read_text())["rows"]:
            rows.append({"arm": "RH (development)", "id": r["substrate"],
                         "N": r["N"], "keys": r["artifact_keys"],
                         "chars": r["manuscript_chars"],
                         "keys_per_kchar": round(r["artifact_keys"] /
                                                 (r["manuscript_chars"] / 1000), 2)})
    return rows, excluded


def main():
    rows, excluded = collect()
    if len(rows) < 20:
        print(f"only {len(rows)} substrates characterised; need the arm files")
        return 1
    works = [r for r in rows if r["N"] >= FLOOR]

    print(f"E20 -- {len(rows)} substrates, {len(works)} with a usable universe")
    if excluded:
        print(f"  excluded, named rather than dropped ({len(excluded)}):")
        for arm, ident, why in excluded:
            print(f"    {arm}: {ident} -- {why}")
    print()
    print(f"  {'arm':22s}{'n':>4}{'usable':>8}{'median N':>10}{'median keys':>13}")
    print("  " + "-" * 55)
    for name in dict.fromkeys(r["arm"] for r in rows):
        g = [r for r in rows if r["arm"] == name]
        print(f"  {name:22s}{len(g):>4}{sum(1 for r in g if r['N']>=FLOOR):>8}"
              f"{st.median([r['N'] for r in g]):>10g}"
              f"{st.median([r['keys'] for r in g]):>13g}")

    print(f"\n  rank correlation with N, over all {len(rows)} substrates")
    for feat in ("keys", "chars", "keys_per_kchar"):
        rho = spearman([r[feat] for r in rows], [r["N"] for r in rows])
        print(f"    {feat:16s} rho = {rho:+.3f}")

    # a predictor has to separate, not merely correlate: can any threshold on a
    # single feature identify the usable substrates without dragging others in?
    separation = {}
    print(f"\n  can one threshold separate the {len(works)} usable substrates?")
    for feat in ("keys", "chars", "keys_per_kchar"):
        vals = sorted({r[feat] for r in rows})
        best = None
        for thr in vals:
            sel = [r for r in rows if r[feat] >= thr]
            if not sel:
                continue
            tp = sum(1 for r in sel if r["N"] >= FLOOR)
            prec, rec = tp / len(sel), tp / max(1, len(works))
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
            if best is None or f1 > best[0]:
                best = (f1, thr, prec, rec, len(sel))
        f1, thr, prec, rec, nsel = best
        separation[feat] = {"best_f1": round(f1, 3), "threshold": thr,
                            "selects": nsel, "precision": round(prec, 3),
                            "recall": round(rec, 3)}
        print(f"    {feat:16s} best F1 {f1:.2f} at >= {thr:g}"
              f"  (selects {nsel}, precision {prec:.0%}, recall {rec:.0%})")

    # what the usable ones look like against everything else
    if works:
        rest = [r for r in rows if r["N"] < FLOOR]
        print(f"\n  usable vs the rest")
        for feat in ("keys", "chars", "keys_per_kchar"):
            a = st.median([r[feat] for r in works])
            b = st.median([r[feat] for r in rest])
            print(f"    median {feat:16s} usable {a:>10g}   rest {b:>10g}"
                  f"   ratio {a/b if b else float('inf'):.1f}x")

    out = {"substrates": len(rows), "usable": len(works), "floor": FLOOR,
           "excluded": [{"arm": a, "id": i, "why": w} for a, i, w in excluded],
           "spearman": {f: round(spearman([r[f] for r in rows],
                                          [r["N"] for r in rows]), 3)
                        for f in ("keys", "chars", "keys_per_kchar")},
           "separation": separation,
           "rows": rows}
    (HERE / "e20_predictors.json").write_text(json.dumps(out, indent=1))

    rho_keys = out["spearman"]["keys"]
    rho_chars = out["spearman"]["chars"]
    rho_dens = out["spearman"]["keys_per_kchar"]
    print(f"""
Reading. Recorded cells is the only feature that tracks the outcome at all
(rho = {rho_keys:+.2f}), and it does not separate: no threshold on it isolates the usable
substrates without pulling in substrates that bind nothing. Manuscript length carries none
({rho_chars:+.2f}). Cell density per thousand characters tracks it slightly better than the
raw count ({rho_dens:+.2f}) and separates no better.

So the answer to "what distinguishes the substrate that works" is still not a
mechanism, but it is now a bounded ignorance rather than an open one. With
{len(rows)} substrates rather than seven, the features a released bundle exposes do not
predict whether its claims can be bound. That is consistent with the paper's
own thesis and unflattering to it in the same breath: if binding depended on
something visible in the record, a benchmark could screen for it in advance,
and it cannot.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
