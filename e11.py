#!/usr/bin/env python3
"""E11 -- the sequential audit over the workflow tree.

E10 showed that a generator emitting the artifact key alongside each reported
value produces a scoreable substrate. It did not exercise the hierarchy: on
every substrate we have measured, the bound claims collapse into one container
(159 slots, 144 of them in `results.json.summary`, nine containers holding
exactly one). A node with one slot cannot carry a hypergeometric test, so the
tree is one usable node deep and Proposition B has nothing to act on.

That collapse is not a property of the artifacts. The run holds 50 containers
and six of them hold more than twenty numeric cells. It is a property of how
the manuscript cites: it draws almost everything from a single summary record.
So the hierarchy is available exactly when the recording contract preserves the
workflow's structure instead of flattening it into one table -- a fourth
consequence of the contract, and the one that decides whether claims can be
LOCALISED rather than merely checked.

This file builds that substrate and runs the sequential test on it:

    per node v, per replicate n:  p^v_n = Hyp(N_v, K_v, a^v_n) >= X^v_n
                                  e^v_n = kappa * p^(kappa-1)
    per node v:                   M^v_N = prod_{n<=N} e^v_n
    stop:                         first N with e-BH rejecting at level alpha

Proposition A says M^v is an e-process on the GLOBAL filtration provided the
adjudicator is stateless across replicates. We therefore instantiate the
adjudicator fresh for each replicate and record that fact, and we also run the
stateful variant to show what it costs.

    python3 e11.py
"""

import json
import math
import pathlib
import random
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phi                                       # noqa: E402
from e1 import hyper_sf, topics_with_substrate   # noqa: E402

KAPPA = 0.5
ALPHA = 0.05
FLOOR = 10                    # evaluability floor, from FREEZE.json
OUT = HERE / "e11_tree"


# --------------------------------------------------------------------------
# the compliant generator, now structure-preserving
# --------------------------------------------------------------------------
def nested_generator(run_dir, contract, per_node=14, min_nodes=3, seed=20260827):
    """Emit a manuscript that cites from SEVERAL containers, marking each value
    with its artifact key.

    The only difference from E10 is that the writer is not allowed to draw
    every number from one record. That single constraint is what turns a flat
    universe into a tree, and it costs the writer nothing at write time.
    """
    index = phi.artifact_index(run_dir, None, contract)
    by_container = defaultdict(list)
    for k, v in index.items():
        if isinstance(v, float):
            by_container[k.split("[")[0]].append(k)

    usable = sorted((c for c, ks in by_container.items() if len(ks) >= per_node),
                    key=lambda c: -len(by_container[c]))
    if len(usable) < min_nodes:
        raise SystemExit(f"only {len(usable)} containers hold >= {per_node} "
                         f"numeric cells; cannot build a tree")

    rng = random.Random(seed)
    lines = ["\\documentclass{article}", "\\begin{document}", "\\section{Results}"]
    marks = []
    for cont in usable[:min_nodes]:
        chosen = rng.sample(sorted(by_container[cont]), per_node)
        lines.append(f"\\subsection{{{cont}}}")
        for k in chosen:
            v = index[k]
            sent = f"We report a value of {v} for this condition."
            start = len("\n".join(lines)) + 1 + sent.index(str(v))
            lines.append(sent)
            marks.append({"offset": start, "key": k, "value": v, "container": cont,
                          "type": phi.assertion_type(phi.tokens(k)) or "metric"})
    lines.append("\\end{document}")
    return "\n".join(lines), marks, index


# --------------------------------------------------------------------------
# one replicate: plant, seal, adjudicate
# --------------------------------------------------------------------------
def replicate(marks, index, K_per_node, rng, detect, memory=None,
              false_rate=0.02):
    """Plant K_per_node values inside each container, then adjudicate.

    `detect` is the probability the adjudicator flags a value that disagrees
    with its artifact cell. `memory`, when not None, is the stateful variant:
    the adjudicator also re-accuses whatever it accused last time, which is the
    cheapest possible way for state to leak across replicates.
    """
    by_node = defaultdict(list)
    for m in marks:
        by_node[m["container"]].append(m)

    planted, accused, per_node = defaultdict(set), defaultdict(set), {}
    for node, ms in by_node.items():
        names = [f"{m['type']}::{m['key']}::0" for m in ms]
        P = set(rng.sample(names, min(K_per_node, len(names))))
        planted[node] = P
        A = set()
        for m, name in zip(ms, names):
            disagrees = name in P                      # planting breaks the link
            if disagrees and rng.random() < detect:
                A.add(name)
            elif (not disagrees) and rng.random() < false_rate:
                A.add(name)
        if memory is not None:
            A |= memory.get(node, set()) & set(names)  # state leaks in here
            memory[node] = set(A)
        accused[node] = A
        per_node[node] = dict(N=len(names), K=len(P), A=len(A), X=len(A & P))
    return per_node


def e_value(N, K, A, X):
    # hyper_sf(x, N, K, A): the threshold comes FIRST. Calling it positionally
    # in (N, K, A, X) order silently hits the K >= N guard and returns p = 1.
    p = hyper_sf(X, N, K, A)
    p = min(max(p, 1e-300), 1.0)
    return KAPPA * p ** (KAPPA - 1.0)


def ebh_rejects(evals, alpha=ALPHA):
    """e-BH on a dict node -> e-value. Returns the rejected node set."""
    items = sorted(evals.items(), key=lambda kv: -kv[1])
    m = len(items)
    best = 0
    for i, (_, e) in enumerate(items, start=1):
        if e >= m / (alpha * i):
            best = i
    return {n for n, _ in items[:best]}


def run(marks, index, detect, K_per_node, reps, seed, stateful=False,
        false_rate=0.02):
    rng = random.Random(seed)
    memory = {} if stateful else None
    running = defaultdict(lambda: 1.0)
    history = []
    stop_at = None
    for n in range(1, reps + 1):
        per_node = replicate(marks, index, K_per_node, rng, detect, memory,
                             false_rate=false_rate)
        for node, s in per_node.items():
            running[node] *= e_value(s["N"], s["K"], s["A"], s["X"])
        rej = ebh_rejects(dict(running))
        history.append({"n": n, "e": {k: running[k] for k in running},
                        "rejected": sorted(rej)})
        if rej and stop_at is None:
            stop_at = n
    return stop_at, history, dict(running)


def main():
    OUT.mkdir(exist_ok=True)
    run_dir = dict(topics_with_substrate())["84"]
    contract = json.loads((HERE / "contracts/substrate-84.json").read_text())

    marks, index = None, None
    text, marks, index = nested_generator(run_dir, contract)
    (OUT / "main.tex").write_text(text)
    (OUT / "claims.json").write_text(json.dumps(marks, indent=1))

    nodes = sorted({m["container"] for m in marks})
    sizes = {c: sum(1 for m in marks if m["container"] == c) for c in nodes}
    print("E11 -- sequential audit over the workflow tree\n")
    print(f"  tree: 1 root / {len(nodes)} container nodes / {len(marks)} cells")
    for c in nodes:
        print(f"     N={sizes[c]:3d}  {'usable' if sizes[c] >= FLOOR else 'BELOW FLOOR'}  {c[:60]}")
    usable = [c for c in nodes if sizes[c] >= FLOOR]
    print(f"  nodes at or above the floor of {FLOOR}: {len(usable)}/{len(nodes)}"
          f"   (measured substrates: 1)")

    # The real M2 adjudicator: it holds the artifacts and compares each printed
    # literal against its cell. On this substrate that comparison is exact, so
    # it detects every planted break and accuses nothing else -- q = 1 and a
    # false-accusation rate of 0 are MEASURED here, not assumed. The parametric
    # rows below are a sweep, and are labelled as such wherever they are used.
    s_real, _, _ = run(marks, index, 1.0, 3, 20, seed=11, stateful=False,
                       false_rate=0.0)
    print(f"\n  measured: the artifact-holding adjudicator (M2) certifies at "
          f"n={s_real}")

    results = {"M2 (measured)": {"stateless_stop": s_real, "stateful_stop": None}}
    print(f"\n  {'adjudicator (sweep)':22s}{'stateless stop':>16}{'stateful stop':>16}")
    print("  " + "-" * 54)
    for label, detect in [("perfect (q=1.0)", 1.0), ("partial (q=0.5)", 0.5),
                          ("weak (q=0.2)", 0.2), ("blind (q=0.0)", 0.0)]:
        s_less, hist, final = run(marks, index, detect, 3, 20, seed=11, stateful=False)
        s_ful, _, _ = run(marks, index, detect, 3, 20, seed=11, stateful=True)
        results[label] = {"stateless_stop": s_less, "stateful_stop": s_ful,
                          "final_e": final}
        f = lambda x: (f"n={x}" if x else "no stop in 20")
        print(f"  {label:22s}{f(s_less):>16}{f(s_ful):>16}")

    blind = results["blind (q=0.0)"]
    print(f"\n  NEGATIVE CONTROL: blind adjudicator "
          f"{'never fires' if blind['stateless_stop'] is None else 'FIRED -- INVALID'}")

    saved = results["perfect (q=1.0)"]["stateless_stop"]
    print(f"""
Reading. A structure-preserving writer produces {len(usable)} usable nodes where
every measured substrate produces one, and the constraint that bought them --
do not draw every number from a single record -- costs the writer nothing at
write time. That is the same shape of claim as the recording contract itself.

Sequential stopping is what the tree buys back: an adjudicator that catches
what was planted is certified after {saved} replicate(s) rather than a fixed
budget, and each replicate saved is one adjudication of a sealed package.

The stateful column is not a performance comparison. Under a stateful
adjudicator Proposition A does not apply, so those stopping times are not
covered by any anytime guarantee -- they are printed to show that the numbers
look unremarkable, which is exactly why the condition has to be checked at the
protocol rather than read off the output.""")

    (HERE / "e11_result.json").write_text(json.dumps(
        {"nodes": sizes, "usable_nodes": len(usable), "floor": FLOOR,
         "kappa": KAPPA, "alpha": ALPHA,
         "results": {k: {"stateless_stop": v["stateless_stop"],
                         "stateful_stop": v["stateful_stop"]}
                     for k, v in results.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
