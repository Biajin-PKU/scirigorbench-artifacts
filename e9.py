#!/usr/bin/env python3
"""E9 -- can the third face be retrofitted?

E5 patched the artifact side of the recording contract and the arm still did not
become scoreable. The face that was missing is the claim side: a marking that
says which artifact CELL each printed number refers to. The obvious next move is
to add that marking too, mechanically, from what the run already holds.

This experiment is the reason that move is not available.

A retrofitted claim marking has only one signal to work from: the printed VALUE.
Nothing else in an unmarked manuscript distinguishes the number that reports
cell A from the number that reports cell B. And a universe derived from values
is a function of the labelling, which the construction forbids for a reason the
acceptance test makes concrete: the slot names must be identical before and
after planting, or the null hypothesis is not about what we claim it is.

So we build the retrofit, run the acceptance test on it, and let it fail.

    python3 e9.py --repo /tmp/mlrb/r
"""

import argparse
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "arms"))
import phi  # noqa: E402
from arms.mlrbench_agent import manuscripts, BACKEND  # noqa: E402

NUM = re.compile(r"(?<![\w.])\d+\.\d+(?![\w])")


def retrofit_marking(text, index, tol=1e-9):
    """The only retrofit available: bind each printed numeral to the cell whose
    value it equals. Returns {position: artifact key}.

    This is the procedure a well-intentioned engineer would write, and it is
    exactly the one the construction rules out.
    """
    by_value = {}
    for k, v in index.items():
        by_value.setdefault(round(float(v), 6), []).append(k)
    marks = {}
    for m in NUM.finditer(text):
        v = round(float(m.group(0)), 6)
        hits = by_value.get(v)
        if hits and len(hits) == 1:
            marks[m.start()] = hits[0]
    return marks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--task", default="iclr2025_scope")
    ap.add_argument("--writer", default="claude-3-7-sonnet-20250219")
    ap.add_argument("--delta", type=float, default=0.30)
    a = ap.parse_args()

    td = pathlib.Path(a.repo) / "agent_results/experiments_and_writeups" / BACKEND / a.task
    contract = json.loads((HERE / "contracts/mlrbench-agent-claude.json").read_text())
    text = manuscripts(td)[a.writer]
    index = phi.artifact_index(td, phi.tokens(text), contract)

    before = retrofit_marking(text, index)
    print(f"task {a.task}, writer {a.writer}")
    print(f"  artifact keys           : {len(index)}")
    print(f"  retrofitted claim marks : {len(before)}   "
          f"(vs N={phi.extract(td, contract, text=text)['N']} from the frame matcher)")

    # Perturb the manuscript the way planting would: move a marked value.
    if not before:
        print("\n  no marks to perturb; retrofit yields nothing on this arm")
        return 0
    pos = sorted(before)[0]
    m = NUM.match(text, pos) or NUM.search(text, pos)
    old = m.group(0)
    new = f"{float(old) * (1 + a.delta):.{len(old.split('.')[1])}f}"
    planted = text[:m.start()] + new + text[m.end():]
    after = retrofit_marking(planted, index)

    print(f"\n  planted one value: {old} -> {new}  (delta={a.delta})")
    print(f"  marks before : {len(before)}")
    print(f"  marks after  : {len(after)}")

    same = set(before.values()) == set(after.values())
    lost = set(before.values()) - set(after.values())
    print(f"\n  ACCEPTANCE TEST -- slot-name set invariant under planting: "
          f"{'PASS' if same else 'FAIL'}")
    if not same:
        print(f"    {len(lost)} slot(s) disappeared from the universe when their "
              f"own value was changed:")
        for k in sorted(lost)[:3]:
            print(f"      {k[:78]}")
        print(f"""
    This is the failure the construction is designed to make visible. The
    retrofitted universe is a function of the values it contains, so planting
    does not perturb a claim inside a fixed universe -- it removes the claim
    from the universe. N, K, A and X are then computed against different
    objects before and after, and the hypergeometric null is not the null we
    say it is.

    The artifact side could be re-serialised because the values were already
    recorded as data (E5). The claim side cannot, because nothing in an
    unmarked manuscript says which cell a number refers to EXCEPT the number.

    The third face of the recording contract is therefore not a patch. It has
    to be emitted when the manuscript is written, and that is why the one arm
    that clears the floor is the one whose manuscript is generated from a macro
    table: there, the marking exists at generation time by construction.""")

    (HERE / "e9_retrofit.json").write_text(json.dumps(
        {"task": a.task, "writer": a.writer, "artifact_keys": len(index),
         "marks_before": len(before), "marks_after": len(after),
         "invariant": same, "lost": sorted(lost)[:10]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
