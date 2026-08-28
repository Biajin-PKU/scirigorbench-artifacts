#!/usr/bin/env python3
"""E10 -- is the prescription sufficient? A reference-compliant generator.

Everything so far establishes that the third face is NECESSARY. Four routes
failed without it: patching the artifact side (E5), retrofitting the claim side
(E9), consuming a claim-to-file pointer (ARA), and consuming a claim-to-quantity
name (macro tables). None of that shows it is ENOUGH.

Sufficiency is not a thing to measure on a corpus that does not comply. It is a
thing to demonstrate with a reference implementation: a generator that emits, at
write time, the artifact key each printed number reports. This file is that
generator, and then the whole pipeline run against what it produced.

The manuscript it writes is deliberately unremarkable prose -- the point is not
that the sentences are good, it is that the pointer travels with them. Values
are read from a real run's artifacts; nothing here invents a number.

    python3 e10.py
"""

import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phi                                     # noqa: E402
from e1 import calibrate, certify, hyper_sf, topics_with_substrate  # noqa: E402

OUT = HERE / "e10_compliant"


def compliant_generator(run_dir, contract, limit=40, seed=20260827):
    """Write a manuscript AND the claim->cell map, together, as a compliant
    agent would.

    The map is produced by the writer because the writer knows what it is
    reporting. That is the whole content of face C, and it is why the face
    cannot be added afterwards: at write time this information is free, and
    after write time it does not exist.
    """
    index = phi.artifact_index(run_dir, None, contract)
    keys = sorted(k for k, v in index.items() if isinstance(v, float))
    rng = random.Random(seed)
    chosen = rng.sample(keys, min(limit, len(keys)))

    lines, marks = ["\\documentclass{article}", "\\begin{document}",
                    "\\section{Results}"], []
    for i, k in enumerate(chosen):
        v = index[k]
        sent = f"We report a value of {v} for this condition."
        # The pointer is emitted with the sentence, keyed by character offset.
        start = len("\n".join(lines)) + 1 + sent.index(str(v))
        lines.append(sent)
        marks.append({"offset": start, "key": k, "value": v,
                      "type": phi.assertion_type(phi.tokens(k)) or "metric"})
    lines.append("\\end{document}")
    return "\n".join(lines), marks


def extract_marked(text, marks, index):
    """Phi's job when face C is present: read the pointer. No frame matching.

    Slot name is (assertion type, artifact key, occurrence) exactly as before,
    so the identity is still value-independent -- the pointer supplies the key
    rather than a text window guessing it.
    """
    slots = {}
    for m in marks:
        if m["key"] not in index:
            continue
        name = f"{m['type']}::{m['key']}::0"
        s = slots.setdefault(name, {"type": m["type"], "artifact_key": m["key"],
                                    "artifact_value": index[m["key"]],
                                    "positions": []})
        lit = str(m["value"])
        s["positions"].append({"start": m["offset"], "end": m["offset"] + len(lit),
                               "literal": lit,
                               "dp": len(lit.split(".")[1]) if "." in lit else 0,
                               "relation": None})
    return slots


def main():
    OUT.mkdir(exist_ok=True)
    topic, run_dir = dict(topics_with_substrate()).get("84") and ("84", dict(topics_with_substrate())["84"])
    contract = json.loads((HERE / "contracts/substrate-84.json").read_text())
    index = phi.artifact_index(run_dir, None, contract)

    text, marks = compliant_generator(run_dir, contract)
    (OUT / "main.tex").write_text(text)
    (OUT / "claims.json").write_text(json.dumps(marks, indent=1))

    slots = extract_marked(text, marks, index)
    N = len(slots)
    print("E10 -- reference-compliant generator\n")
    print(f"  claims emitted      : {len(marks)}")
    print(f"  claims resolved (N) : {N}   ({N / max(1, len(marks)):.0%})")
    print(f"  evaluable (N >= 10) : {'YES' if N >= 10 else 'no'}")

    # --- acceptance test: slot names invariant under planting ----------------
    rng = random.Random(7)
    K = 3
    planted = set(rng.sample(sorted(slots), K))
    text2, marks2 = text, []
    for m in marks:
        name = f"{m['type']}::{m['key']}::0"
        if name in planted:
            new = m["value"] * 1.30
            lit, nlit = str(m["value"]), f"{new:.{len(str(m['value']).split('.')[1]) if '.' in str(m['value']) else 0}f}"
            text2 = text2.replace(f"value of {lit} for", f"value of {nlit} for", 1)
            marks2.append({**m, "value": float(nlit)})
        else:
            marks2.append(m)
    slots2 = extract_marked(text2, marks2, index)
    invariant = set(slots) == set(slots2)
    print(f"\n  ACCEPTANCE TEST -- slot names invariant under planting: "
          f"{'PASS' if invariant else 'FAIL'}")

    # --- the certificate, unchanged -----------------------------------------
    accused = set()
    for name, s in slots2.items():
        dp = min(p["dp"] for p in s["positions"])
        if round(float(s["positions"][0]["literal"]), dp) != round(float(s["artifact_value"]), dp):
            accused.add(name)
    r = certify(list(slots), planted, accused, "M2")
    print(f"  certificate         : N={r['N']} K={r['K']} A={r['A']} X={r['X']} "
          f"p={r['p']:.3g} e={r['e']:.3g}")

    null = certify(list(slots), set(), set(), "M2-null")
    accused0 = {n for n, s in slots.items()
                if round(float(s["positions"][0]["literal"]),
                         min(p["dp"] for p in s["positions"]))
                != round(float(s["artifact_value"]),
                         min(p["dp"] for p in s["positions"]))}
    print(f"  negative control    : accused={len(accused0)}  "
          f"{'OK' if not accused0 else 'FALSE ACCUSATION'}")

    ok = (N >= 10) and invariant and r["A"] == r["K"] == r["X"] and not accused0
    print(f"""
{'SUFFICIENT' if ok else 'NOT SUFFICIENT'}: a generator that emits the artifact key
alongside each reported value produces a substrate on which every claim
resolves, the universe is invariant under planting, the certificate fires
exactly on what was planted, and the negative control is clean.

Necessity was measured on corpora that do not comply (E5, E9, ARA, macro
tables). Sufficiency is this: not a claim about what agents do, but a
demonstration of what the requirement buys when it is met.""")
    (HERE / "e10_result.json").write_text(json.dumps(
        {"emitted": len(marks), "N": N, "invariant": invariant,
         "certificate": r, "null_accusations": len(accused0),
         "sufficient": ok}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
