#!/usr/bin/env python3
"""First external arm: AI Scientist-v2 as released inside MLR-Bench.

The killer question for the leaderboard is not whether we can capture a trace.
It is whether Phi extracts a usable universe from an agent that is not ours --
coverage on the development arm is 43%, and that arm is the one the extractor
was built against. MLR-Bench ships ten v2 runs with BOTH the manuscript and the
run's summary artifacts, MIT licensed, so that question can be answered today
without a GPU, a sandbox or a single agent run.

    chchenhui/mlrbench : ai_scientist_v2_papers/o4-mini/<task>/
        <task>.pdf                       the claim side
        experiments/*_summary.json       the artifact side
        review_*.json                    the arm's own reviewers -- excluded

Two things must travel with every number this produces:

  * The artifacts were captured by the MLR-Bench authors, not by us at a
    uniform sandbox boundary. Whatever cross-arm comparability the boundary
    buys, it does not buy it here.
  * The manuscript is a PDF. On the development arm 324 of 334 numbers lived in
    LaTeX tables read structurally by column spec; a PDF has no column spec, so
    coverage here is not comparable to coverage there even in principle.

Held-out split, fixed before any measurement: iclr2025_dl4c is the DEVELOPMENT
task, because the PDF path cannot be developed on a LaTeX substrate. The other
nine are held out.

    python3 arms/mlrbench_v2.py --repo /tmp/mlrb/r --task iclr2025_dl4c
    python3 arms/mlrbench_v2.py --repo /tmp/mlrb/r --all
"""

import argparse
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import phi  # noqa: E402

DEV_TASK = "iclr2025_dl4c"
CONTRACT = pathlib.Path(__file__).resolve().parent.parent / \
    "contracts/mlrbench-v2-o4mini.json"


import re

# Furniture a review-formatted PDF carries that asserts nothing about the
# research. Each of these was producing FALSE CLAIMS on the development task:
# the extractor read the margin line numbers as numeric assertions and typed
# them `stat_test`, which is how a paper with a handful of real numbers
# produced 210 "claims".
MARGIN_LINE_NO = re.compile(r"^[ \t]*\d{1,4}[ \t]*$", re.M)
RUNNING_HEADER = re.compile(r"^[ \t]*Under review as a .*$", re.M)
PAGE_NUMBER = re.compile(r"^[ \t]*\d{1,3}[ \t]*$", re.M)
# pdftotext splits LaTeX small-caps: "T RACE C ODE", "R EPRESENTATIONS".
# Left alone it corrupts the frame tokens a claim is resolved by -- the
# development task produced `epresentations` as a token.
SMALLCAPS = re.compile(r"\b([A-Z]) ([A-Z]{2,})")


# A citation year is not a claim. The LaTeX path never sees one, because
# PLUMBING_RE strips \cite/\citep/\citet before extraction; a PDF has already
# rendered them into "(Author et al., 2018)". Masking these is parity with the
# LaTeX path, not a new heuristic.
CITATION = re.compile(r"\((?:[^()]*?,\s*)?(?:19|20)\d{2}[a-z]?\)")
CITATION_BARE = re.compile(r"(?<=,\s)(?:19|20)\d{2}[a-z]?(?=[);])")


# The bibliography block. Page ranges, volumes and arXiv ids are numerals that
# assert nothing about the research, and the LaTeX path never sees them because
# PLUMBING_RE strips the citation commands. Cut the BLOCK, not the tail: this
# template puts SUPPLEMENTARY MATERIAL after the references, and that appendix
# carries real implementation claims.
BIB_BLOCK = re.compile(
    r"^[ \t]*R\s?EFERENCES?\b.*?(?=^[ \t]*(?:S\s?UPPLEMENTARY|A\s?PPENDIX|"
    r"[A-Z]\.?\s+[A-Z]{2,})|\Z)", re.M | re.S)

MIN_COLUMN_RUN = 10          # frozen: how many aligned integers make a column


def _line_number_column(words, x_tol=1.0):
    """ids of words belonging to a review template's line-number column.

    The column is FOUND, not assumed. A hard-coded margin fraction was tried
    first and silently removed nothing: the numbers sit at x=73.1 on a 612pt
    page, and 0.11 of the width is 67.3. Guessing the constant is exactly the
    class of mistake the artifact-scope work already paid for once.

    A column is a set of bare integers sharing an x position, numerous enough
    not to be a coincidence, and increasing down the page -- which a table's
    left-hand column of measurements is not.
    """
    by_x = {}
    for w in words:
        if not w[4].strip().isdigit():
            continue
        key = round(w[0] / x_tol) * x_tol
        by_x.setdefault(key, []).append(w)
    drop = set()
    for _, ws in by_x.items():
        if len(ws) < MIN_COLUMN_RUN:
            continue
        ordered = sorted(ws, key=lambda w: w[1])
        vals = [int(w[4]) for w in ordered]
        if all(b > a for a, b in zip(vals, vals[1:])):
            drop |= {id(w) for w in ordered}
    return drop


def _page_number_words(words, rect, bottom_frac=0.94):
    """ids of bare integers in the page's bottom strip: folios, not claims."""
    y_cut = rect.y0 + bottom_frac * rect.height
    return {id(w) for w in words if w[1] > y_cut and w[4].strip().isdigit()}


def manuscript_text(task_dir, raw=False):
    """PDF -> text with the review furniture removed structurally.

    The margin line numbers of a review-formatted paper are a narrow column at
    a fixed x-offset, and that is how they are removed: by geometry, from
    PyMuPDF word boxes. Lexical rules were tried first and were not adequate --
    `-layout` puts some line numbers on their own line and injects others
    mid-sentence, so a line-anchored pattern removed 80 of them and left 50
    behind, still typed as numeric assertions. This mirrors the decision made
    for tables on the LaTeX path: read the structure, not a character window.

    `margin_frac` is the fraction of page width treated as margin. It is a
    threshold and it is frozen with the rest of the instrument.
    """
    pdfs = sorted(task_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDF in {task_dir}")
    if raw:
        return subprocess.run(["pdftotext", "-layout", str(pdfs[0]), "-"],
                              capture_output=True, text=True, check=True).stdout

    import fitz
    doc = fitz.open(str(pdfs[0]))
    out = []
    for page in doc:
        words = [w for w in page.get_text("words")]     # x0,y0,x1,y1,word,...
        drop = _line_number_column(words) | _page_number_words(words, page.rect)
        kept = [w for w in words if id(w) not in drop]
        kept.sort(key=lambda w: (round(w[1], 1), w[0]))
        line, y, lines = [], None, []
        for w in kept:
            if y is None or abs(w[1] - y) > 3:
                if line:
                    lines.append(" ".join(line))
                line, y = [], w[1]
            line.append(w[4])
        if line:
            lines.append(" ".join(line))
        out.append("\n".join(lines))
    t = "\n".join(out)
    doc.close()

    t = RUNNING_HEADER.sub("", t)
    t = SMALLCAPS.sub(lambda m: m.group(1) + m.group(2), t)
    t = CITATION.sub(" ", t)
    t = CITATION_BARE.sub(" ", t)
    t = BIB_BLOCK.sub("\n", t)
    return t


def run_task(task_dir, contract):
    text = manuscript_text(task_dir)
    uni = phi.extract(task_dir, contract, text=text)
    st = phi.strata(uni["slots"])
    return {
        "task": task_dir.name,
        "manuscript_chars": len(text),
        "artifact_keys": uni["artifact_keys"],
        "N": uni["N"],
        "unresolved": uni["unresolved_claims"],
        "by_cause": uni["unresolved_by_cause"],
        "parse_failure_rate": uni["parse_failure_rate"],
        "not_recorded_rate": uni["not_recorded_rate"],
        "unreadable_declared": len(uni["unreadable_declared_files"]),
        "N_by_layer": {k: len(v) for k, v in st.items() if v},
        "universe_digest": phi.universe_digest(uni)[:16],
        "evaluable": uni["N"] >= 10,      # D3 section 3
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--task")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="arms/mlrbench_v2_phi.json")
    a = ap.parse_args()

    root = pathlib.Path(a.repo) / "ai_scientist_v2_papers/o4-mini"
    contract = json.loads(CONTRACT.read_text())
    tasks = sorted(p for p in root.iterdir() if p.is_dir())
    if a.task:
        tasks = [t for t in tasks if t.name == a.task]
    elif not a.all:
        tasks = [t for t in tasks if t.name == DEV_TASK]

    rows = []
    hdr = f"{'task':24s} {'keys':>7} {'N':>5} {'unres':>6} {'ours':>7} {'theirs':>7} {'eval':>5}"
    print(hdr); print("-" * len(hdr))
    for t in tasks:
        r = run_task(t, contract)
        rows.append(r)
        held = "" if r["task"] == DEV_TASK else ""
        print(f"{r['task']:24s} {r['artifact_keys']:>7} {r['N']:>5} "
              f"{r['unresolved']:>6} {r['parse_failure_rate']:>6.1%} "
              f"{r['not_recorded_rate']:>6.1%} {'YES' if r['evaluable'] else 'NO':>5}{held}")
        pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))

    if len(rows) > 1:
        ev = sum(1 for r in rows if r["evaluable"])
        Ns = sorted(r["N"] for r in rows)
        print(f"\nevaluable (N>=10): {ev}/{len(rows)}   N median={Ns[len(Ns)//2]}  "
              f"range={Ns[0]}-{Ns[-1]}")
        print(f"held-out only    : "
              f"{sum(1 for r in rows if r['evaluable'] and r['task'] != DEV_TASK)}"
              f"/{sum(1 for r in rows if r['task'] != DEV_TASK)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
