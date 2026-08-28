#!/usr/bin/env python3
"""ICML 2026 agent reproduction challenge as a fifth arm.

This corpus is different in kind from the other four. It is not one agent's
release but a community challenge, and its submissions carry an EXPLICIT claim
structure: a logbook indexes one page per claim, and the claim titles state the
numbers. That is a face-C implementation in the wild, like ARA's, and it is the
second chance to consume one rather than describe it.

Per submission:

    pages/**/page.md   the claim pages -- the manuscript side
    logbook.json       the claim index
    artifacts/*        whatever machine-readable results were published

    python3 arms/icml_repro.py --repo /tmp/icml_repro
"""

import argparse
import json
import pathlib
import statistics as st
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import phi  # noqa: E402


def manuscript(sub):
    """The claim pages, in logbook order when the index is present."""
    lb = sub / "logbook.json"
    parts = []
    if lb.exists():
        try:
            d = json.loads(lb.read_text())
            def walk(node):
                f = node.get("file")
                if f and (sub / f).exists():
                    parts.append((sub / f).read_text(errors="ignore"))
                for c in node.get("children", []):
                    walk(c)
            walk(d.get("root", {}))
        except Exception:
            pass
    if not parts:
        parts = [p.read_text(errors="ignore")
                 for p in sorted(sub.glob("pages/**/*.md"))]
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/tmp/icml_repro")
    ap.add_argument("--out", default="arms/icml_repro_phi.json")
    a = ap.parse_args()

    root = pathlib.Path(a.repo)
    contract = json.loads((HERE / "contracts/icml-repro-challenge.json").read_text())
    subs = sorted(p for p in root.iterdir() if p.is_dir())

    hdr = (f"{'submission':46s}{'chars':>8}{'keys':>7}{'N':>5}{'parse%':>8}{'notrec%':>9}")
    print(f"ICML 2026 reproduction challenge -- {len(subs)} submissions with artifacts\n")
    print(hdr); print("-" * len(hdr))

    rows = []
    for sub in subs:
        text = manuscript(sub)
        idx = phi.artifact_index(sub, None, contract)
        if not text:
            print(f"{sub.name[:44]:46s}  [no claim pages published]")
            rows.append({"submission": sub.name, "manuscript_chars": 0,
                         "artifact_keys": len(idx), "N": 0, "no_claims": True})
            continue
        res = phi.extract(sub, contract, text=text)
        r = {"submission": sub.name, "manuscript_chars": len(text),
             "artifact_keys": len(idx), "N": res["N"],
             "unresolved": res.get("unresolved", 0),
             "parse_failure_rate": res.get("parse_failure_rate", 0.0),
             "not_recorded_rate": res.get("not_recorded_rate", 0.0),
             "by_cause": res.get("unresolved_by_cause", {}),
             "evaluable": res["N"] >= 10}
        rows.append(r)
        print(f"{sub.name[:44]:46s}{len(text):>8}{len(idx):>7}{res['N']:>5}"
              f"{r['parse_failure_rate']*100:>7.1f}%{r['not_recorded_rate']*100:>8.1f}%")
        pathlib.Path(HERE / a.out).write_text(json.dumps(rows, indent=1))

    scored = [r for r in rows if not r.get("no_claims")]
    if scored:
        Ns = [r["N"] for r in scored]
        print(f"\n  median N {st.median(Ns):g}   max {max(Ns)}   "
              f"evaluable {sum(1 for r in scored if r['evaluable'])}/{len(scored)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
