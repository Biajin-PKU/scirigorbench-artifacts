#!/usr/bin/env python3
"""AI Scientist v1 as a fourth arm, on a corpus independent of MLR-Bench.

Two things make this arm worth having. It is a different agent from the v2 line
already audited, and it releases the manuscript as LATEX SOURCE rather than a
PDF -- which is the format our matcher handles best, so a low N here cannot be
blamed on PDF extraction the way it can on the v2 arm.

Structure per released paper:

    latex/template.tex     the manuscript, as the agent wrote it
    run_N/final_info.json  nested per-dataset metric summaries

    python3 arms/aiscientist_v1.py --repo /tmp/aisci_v1
"""

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import phi  # noqa: E402


def manuscript(paper_dir):
    tex = paper_dir / "latex" / "template.tex"
    return tex.read_text(errors="ignore") if tex.exists() else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/tmp/aisci_v1")
    ap.add_argument("--out", default="arms/aiscientist_v1_phi.json")
    a = ap.parse_args()

    root = pathlib.Path(a.repo) / "example_papers"
    contract = json.loads((HERE / "contracts/aiscientist-v1.json").read_text())
    papers = sorted(p for p in root.iterdir() if p.is_dir())

    hdr = (f"{'paper':32s}{'chars':>8}{'keys':>7}{'N':>5}{'unres':>7}"
           f"{'parse%':>8}{'notrec%':>9}")
    print(f"AI Scientist v1 -- {len(papers)} released papers\n")
    print(hdr); print("-" * len(hdr))

    rows = []
    for pd in papers:
        text = manuscript(pd)
        if not text:
            print(f"{pd.name:32s}  [no latex/template.tex]")
            continue
        res = phi.extract(pd, contract, text=text)
        idx = phi.artifact_index(pd, None, contract)
        r = {"paper": pd.name, "manuscript_chars": len(text),
             "artifact_keys": len(idx), "N": res["N"],
             "unresolved": res.get("unresolved", 0),
             "parse_failure_rate": res.get("parse_failure_rate", 0.0),
             "not_recorded_rate": res.get("not_recorded_rate", 0.0),
             "by_cause": res.get("unresolved_by_cause", {}),
             "evaluable": res["N"] >= 10,
             "universe_digest": res.get("universe_digest")}
        rows.append(r)
        print(f"{pd.name:32s}{len(text):>8}{len(idx):>7}{res['N']:>5}"
              f"{r['unresolved']:>7}{r['parse_failure_rate']*100:>7.1f}%"
              f"{r['not_recorded_rate']*100:>8.1f}%")
        pathlib.Path(HERE / a.out).write_text(json.dumps(rows, indent=1))

    import statistics as st
    if rows:
        Ns = [r["N"] for r in rows]
        print(f"\n  median N {st.median(Ns):g}   max {max(Ns)}   "
              f"evaluable {sum(1 for r in rows if r['evaluable'])}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
