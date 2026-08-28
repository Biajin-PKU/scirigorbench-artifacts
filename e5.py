#!/usr/bin/env python3
"""E5 -- the recording patch: is D3 a prescription or a complaint?

The first external arm returned nothing scoreable. The recording contract says
that is fixable by recording differently, not by evaluating differently. E5 is
the constructive form of that claim: take the arm that could not be scored, add
the minimum D3-shaped record, and see whether it crosses the evaluability floor.

The record is a RE-SERIALISATION, never an invention. Every value written into
it is already present in the released run -- the final point of a curve the run
logged, a literal in the script it shipped, a count in its own token ledger. If
a value is not in the run, it is not in the patch, because a patch that supplies
missing numbers would prove that we can write a record, not that the agent could
have.

    python3 e5.py --repo /tmp/mlrb/r
"""

import argparse
import ast
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "arms"))
import phi  # noqa: E402
from arms.mlrbench_v2 import manuscript_text  # noqa: E402

PATCH_NAME = "d3_record.json"


def final(series):
    """Last logged point of a curve. A paper reporting `accuracy' after training
    means the value at the end, and that is the aggregate the curve implies."""
    vals = [v for v in series if isinstance(v, (int, float))]
    return float(vals[-1]) if vals else None


def build_record(task_dir):
    """A D3-shaped record assembled from what this run already contains."""
    rec, src = {}, {}

    # metric: the endpoint of each logged curve, named by the run's own label.
    summary = task_dir / "experiments/research_summary.json"
    if summary.exists():
        d = json.loads(summary.read_text())
        node = d.get("best node", {})
        for m in node.get("metric", {}).get("value", {}).get("metric_names", []):
            name = str(m.get("metric_name", "")).strip()
            data = m.get("data")
            pts = []
            def walk(o):
                if isinstance(o, dict):
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
                elif isinstance(o, (int, float)) and not isinstance(o, bool):
                    pts.append(o)
            walk(data)
            v = final(pts)
            if name and v is not None:
                rec[f"final {name}"] = v
                src[f"final {name}"] = "endpoint of the run's own logged curve"

    # hyperparam / data_source: literals the shipped script actually passes.
    for py in sorted((task_dir / "experiments").glob("best_solution_*.py")):
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if fn in phi.PRESENTATION_CALLS:
                    continue
                for kw in node.keywords:
                    if kw.arg and isinstance(kw.value, ast.Constant) \
                            and isinstance(kw.value.value, (int, float)) \
                            and not isinstance(kw.value.value, bool):
                        rec.setdefault(kw.arg, float(kw.value.value))
                        src.setdefault(kw.arg, f"literal passed in {py.name}")
        break            # both scripts agree; one is the record

    # compute: the run's own ledger.
    tok = task_dir / "token_tracker.json"
    if tok.exists():
        for model, blob in json.loads(tok.read_text()).items():
            for k, v in (blob.get("tokens") or {}).items():
                rec[f"{k} tokens"] = float(v)
                src[f"{k} tokens"] = "the run's own token ledger"
            if "cost (USD)" in blob:
                rec["cost"] = float(blob["cost (USD)"])
                src["cost"] = "the run's own token ledger"
    return rec, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="e5_patch.json")
    a = ap.parse_args()

    root = pathlib.Path(a.repo) / "ai_scientist_v2_papers/o4-mini"
    base_contract = json.loads(
        (HERE / "contracts/mlrbench-v2-o4mini.json").read_text())
    patched = json.loads(json.dumps(base_contract))
    patched["reportable_outputs"] = base_contract["reportable_outputs"] + [PATCH_NAME]
    patched["_patch"] = (
        "E5: a D3-shaped record re-serialised from artifacts the run already "
        "holds. No value originates here.")

    rows = []
    hdr = f"{'task':24s} {'N before':>9} {'N after':>8} {'patch keys':>11} {'crosses 10':>11}"
    print(hdr); print("-" * len(hdr))
    for td in sorted(p for p in root.iterdir() if p.is_dir()):
        text = manuscript_text(td)
        before = phi.extract(td, base_contract, text=text)["N"]

        rec, src = build_record(td)
        (td / PATCH_NAME).write_text(json.dumps(rec, indent=1))
        after = phi.extract(td, patched, text=text)["N"]

        rows.append({"task": td.name, "N_before": before, "N_after": after,
                     "patch_keys": len(rec), "crosses": after >= 10,
                     "provenance": src})
        print(f"{td.name:24s} {before:>9} {after:>8} {len(rec):>11} "
              f"{'YES' if after >= 10 else 'no':>11}")

    ok = sum(1 for r in rows if r["crosses"])
    was = sum(1 for r in rows if r["N_before"] >= 10)
    print(f"\nevaluable before patch: {was}/{len(rows)}"
          f"   after patch: {ok}/{len(rows)}")
    print("Every patched value is a re-serialisation of something the run "
          "already recorded;\nnone of them is supplied by us.")
    pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
