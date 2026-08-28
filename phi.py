#!/usr/bin/env python3
"""The declared-process universe: a deterministic claim/artifact extractor and planter.

This is the instrument the leaderboard is built on.  It answers, for one frozen
run of one agent, "which sentences in the manuscript assert something that the
run's own artifacts can be checked against", and it can plant a known number of
fabrications into exactly those sentences without touching the artifacts.

WHY NOT REUSE THE A-PAPER EXTRACTOR
    reviewer-cert/data/slots_build.py reads `facts.tex`, a macro table the
    generator emits.  The macro NAME is the slot, so slot identity is free and
    value-independent by construction.  External agents do not emit one: they
    print numbers inline.  This file recovers the same guarantee without the
    macro table.  The A-paper substrate is the degenerate case of this one --
    an agent that hands you the artifact key for every printed number.

SLOT IDENTITY  (the premise the certificate rests on)
    A slot is named  (assertion_type, artifact_key, occurrence)  -- never by the
    value it prints.  `artifact_key` is a path into the frozen artifact index,
    and list elements are keyed by their own descriptive string fields rather
    than by position, so the key survives reordering:

        results.json:summary[MOOCCubeX|chronological|DKVMN|ndcg10].mean

    Text -> key resolution is by FRAME tokens (the words around the numeral),
    never by the numeral.  An extractor that grouped occurrences by their
    printed value would make the universe a function of the values, which is
    precisely what the theory forbids.

OCCURRENCES
    One claim is usually printed several times -- abstract, results prose,
    table.  Those are one slot with several positions, not several slots.  This
    matters for planting: changing one printed copy and not the others produces
    an internally inconsistent manuscript, and an auditor that catches it has
    caught a typo, not a fabrication.  The null hypothesis would be testing
    something other than what we claim.

PRECISION IS A PER-SLOT FLOOR, NOT A PER-POSITION ONE
    The abstract may print 82.3 where the table prints 82.34.  The perturbation
    is applied once to the underlying value and re-rendered at each position's
    own precision, so the coarse rendering stays the rounding of the fine one
    and the copies remain mutually consistent.  delta must be large enough to
    move the COARSEST position, otherwise the planting is invisible there.  That
    floor is a property of the manuscript, not a knob, and it bounds how
    concealed a planted claim can be.

WHAT THIS FILE DELIBERATELY DOES NOT DO
    No LLM anywhere.  A universe an LLM samples cannot be pre-registered, and
    without pre-registration the error rates have nothing to attach to.
    Everything here is a published catalog times a dumb matcher.

Usage:
    python3 phi.py --self-check                # acceptance test, needs substrate
    python3 phi.py --run <submission_dir>      # inventory one run
    python3 phi.py --plant <submission_dir> --k 3 --delta 0.30 --seed 20260825
"""

import argparse
import csv
import hashlib
import json
import pathlib
import random
import re
import sys

NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
FRAME_CHARS = 90          # how far around a numeral we read for context
MIN_OVERLAP = 2           # frame tokens that must match a key before we bind it
# Keys derived from released source: `<file>.py:<name>` or `<file>.py:<call>.<kw>`.
CODE_KEY_RE = re.compile(r"\.py:")

# Assertion types that describe a SETTING of the run rather than an OUTCOME of
# it. The catalogue is the organizing principle of the construction, so it is
# also what decides which artifacts a claim may be resolved against: an
# assertion about the batch size has no business resolving to a per-epoch
# accuracy curve, and letting it try is how the highest-scoring irrelevant key
# wins. This mirrors the existing role filter, which already refuses to let a
# centre claim resolve to a dispersion leaf.
SETTING_TYPES = {"hyperparam", "seeds", "compute", "data_source", "split"}

# Presentation calls, whose numeric keywords assert nothing about the research.
# This is the code-side counterpart of PLUMBING_RE: a declared list of surfaces
# whose numbers are about rendering. Without it `xticks(rotation=45)` and
# `text(fontsize=10)` enter the artifact index, and the negative control caught
# exactly that -- a claim resolving to a font size is a false accusation waiting
# to happen, and a benchmark selling a false-accusation rate does not get one.
PRESENTATION_CALLS = frozenset({
    "plot", "scatter", "bar", "barh", "hist", "boxplot", "imshow", "pie",
    "errorbar", "fill_between", "axhline", "axvline", "annotate", "text",
    "xticks", "yticks", "xlabel", "ylabel", "title", "suptitle", "legend",
    "figure", "subplots", "subplot", "subplots_adjust", "tight_layout",
    "savefig", "set_xlim", "set_ylim", "set_title", "set_xlabel", "set_ylabel",
    "grid", "colorbar", "heatmap", "despine", "set_context", "set_theme",
})

# LaTeX plumbing whose numbers assert nothing about the research.
# Rendered citation years. The LaTeX path never sees one because PLUMBING_RE
# strips the citation command; a Markdown or PDF manuscript has already rendered
# it. Masking here is parity across the three claim paths, not a new rule.
CITATION_YEAR_RE = re.compile(
    r"\((?:[^()]{0,80}?,\s*)?(?:19|20)\d{2}[a-z]?\)|(?<=[A-Za-z ])(?:19|20)\d{2}(?=\s*[);,\]])")

PLUMBING_RE = re.compile(
    r"\\(?:usepackage|documentclass|includegraphics|vspace|hspace|setlength|"
    r"scalebox|resizebox|columnwidth|textwidth|cmidrule|multicolumn|multirow|"
    r"label|ref|cite|citep|citet|begin|end|section|subsection)[^\n]*"
)

# ---------------------------------------------------------------------------
# The assertion-type catalog -- deliverable D3, face A.  See
# D3-recording-contract.md for the full contract and the artifact-scope face.
#
# The scale is NOT ours.  Each type is the mechanically-checkable residue of a
# verbatim item from a published, community-adopted reporting checklist:
#
#   [MLRC] The Machine Learning Reproducibility Checklist v2.0, Apr. 7 2020
#          (Pineau et al., arXiv:2003.12206)
#   [NPC]  NeurIPS Paper Checklist, neurips.cc/public/guides/PaperChecklist
#
# Inducing the catalog from RH's own traces would only ever measure what RH
# happens to log, and every other agent records something RH does not.
#
# Out of scope by construction: proofs, assumptions, dependencies, code, README.
# Not numbers, so there is no artifact-side value to compare them against.
# ---------------------------------------------------------------------------
CATALOG = [
    {"id": "metric",
     "norm": "[MLRC] A clear definition of the specific measure or statistics "
             "used to report results.",
     "cues": ("accuracy", "auc", "f1", "ndcg", "recall", "precision", "rmse",
              "mae", "ece", "bleu", "score", "gain", "reduction", "improvement")},
    {"id": "stat_test",
     "norm": "[MLRC] A description of results with central tendency (e.g. mean) "
             "& variation (e.g. error bars). / [NPC] Q6: Does the paper report "
             "error bars suitably and correctly defined or other appropriate "
             "information about the statistical significance of the experiments?",
     "cues": ("p", "pvalue", "significance", "ci", "confidence", "interval",
              "sd", "stderr", "bootstrap")},
    {"id": "seeds",
     "norm": "[MLRC] The exact number of training and evaluation runs.",
     "cues": ("seed", "seeds", "repeat", "repetition", "trial", "trials", "runs")},
    {"id": "split",
     "norm": "[MLRC] The details of train / validation / test splits. / [NPC] Q7: "
             "did you specify all the training details (e.g., data splits, "
             "hyperparameters, how they were chosen)?",
     "cues": ("train", "test", "validation", "val", "split", "fold", "holdout",
              "chronological")},
    {"id": "hyperparam",
     "norm": "[MLRC] The range of hyper-parameters considered, method to select "
             "the best hyper-parameter configuration, and specification of all "
             "hyper-parameters used to generate results.",
     "cues": ("epoch", "epochs", "batch", "lr", "learning", "rate", "dropout",
              "hidden", "layers", "dim", "temperature", "steps")},
    {"id": "data_source",
     "norm": "[MLRC] The relevant statistics, such as number of examples.",
     "cues": ("dataset", "samples", "instances", "users", "items", "records",
              "size", "corpus")},
    {"id": "compute",
     "norm": "[MLRC] The average runtime for each result, or estimated energy "
             "cost. / A description of the computing infrastructure used. / "
             "[NPC] Q8: sufficient information on the computer resources (type "
             "of compute workers, memory, time of execution).",
     "cues": ("gpu", "hours", "minutes", "seconds", "memory", "params",
              "parameters", "flops", "cost")},

    # --- literature stage, anchored on a third published standard ------------
    # The two ML checklists above say nothing about the literature stage, which
    # left four of the six adopted research stages with no auditable claims at
    # all and the layer axis with two levels for every arm.  The remedy is not
    # to invent assertion types -- that would make the scale ours, which is the
    # one thing D3 face A forbids -- but to anchor on a standard that already
    # governs this stage.  PRISMA's flow diagram is exactly a set of counts a
    # report states and a retrieval log can be checked against.
    #
    #   [PRISMA] Page MJ et al., "The PRISMA 2020 statement: an updated
    #            guideline for reporting systematic reviews", BMJ 2021;372:n71.
    #
    # Placed last on purpose: CATALOG order IS the published priority, so
    # appending cannot re-assign a claim that an earlier type already binds.
    {"id": "retrieval",
     "norm": "[PRISMA] Flow diagram: records identified from databases and "
             "registers (n); records screened (n); reports sought for "
             "retrieval (n); reports assessed for eligibility (n).",
     "cues": ("retrieved", "identified", "screened", "search", "searched",
              "queries", "candidates", "citations", "references", "papers",
              "studies", "abstracts", "duplicates")},
    {"id": "inclusion",
     "norm": "[PRISMA] Flow diagram: records removed before screening (n); "
             "records excluded (n); reports excluded with reasons (n); "
             "studies included in review (n).",
     "cues": ("included", "excluded", "eligible", "eligibility", "shortlisted",
              "selected", "ingested", "deduplicated")},
]
CUE_TO_TYPE = {c: t["id"] for t in CATALOG for c in t["cues"]}

# ---------------------------------------------------------------------------
# Stratification.  The stages are ADOPTED verbatim from the six-stage
# decomposition of arXiv:2510.23045, A Survey of AI Scientists -- published,
# neutral, citable.  Inventing our own stages would make the strata a property
# of this benchmark rather than of doing research, which is the one thing the
# layer axis is supposed to avoid.
#
# An assertion belongs to the stage it SPEAKS ABOUT, not the stage of the node
# that emitted it: a metric printed in the writing stage still asserts something
# about execution.
#
# ⚠️ Two of the six stages carry every assertion type and four carry none.  That
# is not an oversight to be patched by inventing types: D3 face A is the
# mechanically-checkable NUMERIC residue of ML reporting checklists, and
# literature review, ideation and writing do not put claim/artifact number pairs
# on the table.  It has to be reported, because it bounds column 4 -- "first
# stage to break" can only ever name a stage that has slots in it.
# ---------------------------------------------------------------------------
LAYERS = (
    "literature_review",
    "idea_generation",
    "experimental_preparation",
    "experimental_execution",
    "scientific_writing",
    "paper_generation",
)
TYPE_TO_LAYER = {
    "retrieval": "literature_review",
    "inclusion": "literature_review",
    "data_source": "experimental_preparation",
    "split": "experimental_preparation",
    "hyperparam": "experimental_preparation",
    "seeds": "experimental_execution",
    "compute": "experimental_execution",
    "metric": "experimental_execution",
    "stat_test": "experimental_execution",
}


def layer_of(slot_name):
    """Stage a slot belongs to. Slot names begin with the assertion type."""
    return TYPE_TO_LAYER.get(slot_name.split("::", 1)[0])


def strata(slots):
    """{stage: sorted slot names}, in the adopted stage order, empty stages kept.

    Empty strata are returned rather than dropped: a stage with no auditable
    claims is a reportable fact about coverage, and silently omitting it would
    let a two-stage universe read as a six-stage one.
    """
    out = {L: [] for L in LAYERS}
    for n in sorted(slots):
        L = layer_of(n)
        if L:
            out[L].append(n)
    return out


SUBTOKEN_RE = re.compile(r"[A-Za-z]+|\d+")

# `p < 0.001` asserts a BOUND, not a value.  The artifact holding 0.0 satisfies
# it.  An adjudicator that compares such a claim by equality manufactures a
# false accusation -- our own K=0 negative control caught ours doing exactly
# that, ten times on one run.  A benchmark whose selling point is the false-
# accusation rate does not get to have one.
RELATION_RE = re.compile(r"(<=|>=|<|>|\\leq|\\geq|\\le|\\ge|≤|≥)\s*\$?\s*$")
RELATION_MAP = {"<": "lt", "\\leq": "le", "\\le": "le", "≤": "le", "<=": "le",
                ">": "gt", "\\geq": "ge", "\\ge": "ge", "≥": "ge", ">=": "ge"}

# The two roles a number can play in the `0.458 $\pm$ 0.003` idiom.  Without
# this, every cell ties between .mean/.sd/.ci95_lo/.ci95_hi of the same record
# and resolves to nothing -- the frame genuinely does not say which one it is,
# but the cell's own structure does.
CENTER_LEAVES = {"mean", "value", "estimate", "median", "avg", "score"}
DISPERSION_LEAVES = {"sd", "std", "stderr", "stddev", "se", "sem", "err"}


def assertion_type(frame_toks):
    """First CATALOG entry whose cues the frame hits.  Catalog order IS the
    published priority.

    Iterating the frame's token SET instead would make the answer depend on
    hash order: a results cell framed `MOOCCubeX chronological NDCG` hits both a
    split cue and a metric cue, and which one won varied between runs.  A
    universe that is not reproducible cannot be pre-registered, and without
    pre-registration the error rates attach to nothing.
    """
    for t in CATALOG:
        if frame_toks & set(t["cues"]):
            return t["id"]
    return None


def tokens(text):
    """Words, never bare numerals.

    A standalone numeral is a VALUE, and letting values into the token set would
    let the printed numbers decide which artifact a claim binds to -- the exact
    value-dependence the theory forbids.  WORD_RE is word-initial, so bare
    numerals never enter.  Digits glued to a word are part of a NAME, not a
    value, so `ndcg10` also yields `ndcg` and `10`: that is what lets a column
    headed NDCG@10 meet an artifact key spelled ndcg10.
    """
    out = set()
    for w in WORD_RE.findall(text):
        out.add(w.lower())
        out.update(p.lower() for p in SUBTOKEN_RE.findall(w))
    return out


# ---------------------------------------------------------------------------
# artifact side
# ---------------------------------------------------------------------------

def _descriptor(node):
    """Key a list element by its own IDENTIFIER fields, so the key survives reordering.

    Identifier, not prose: a record that carries a free-text field ("the caption
    compresses panels c and d...") would otherwise put a paragraph of English
    into the key, and that key then out-matches every real one because the
    binder scores on shared words.  Short, few-token strings only.
    """
    if isinstance(node, dict):
        parts = [v for v in node.values()
                 if isinstance(v, str) and len(v) <= 40 and len(v.split()) <= 4]
        if parts:
            return "|".join(parts)
    return None


def _walk_json(node, path, out, nameable=None):
    if isinstance(node, dict):
        for k, v in node.items():
            _walk_json(v, f"{path}.{k}", out, nameable)
    elif isinstance(node, list):
        for v in node:
            name = _descriptor(v)
            if name is None or not _citable(name, nameable):
                continue          # see artifact_index()
            _walk_json(v, f"{path}[{name}]", out, nameable)
    elif isinstance(node, bool):
        pass
    elif isinstance(node, (int, float)):
        out[path] = float(node)


def _walk_csv(path, out, nameable=None):
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return
    text_cols = [c for c in rows[0] if all(not _is_num(r[c]) for r in rows)]
    if not text_cols:
        return                    # unnameable rows: see artifact_index()
    for row in rows:
        rid = "|".join(str(row[c]) for c in text_cols)
        if not _citable(rid, nameable):
            continue
        for col, cell in row.items():
            if _is_num(cell):
                out[f"{path.name}[{rid}].{col}"] = float(cell)


def _citable(name, nameable):
    """Can the manuscript name this row -- ALL of it, not just part of it?

    A predictions row is keyed `MOOCCubeX|chronological|DKVMN|full|Q0-000-12|test`.
    Three of those words are in the paper and one, the query id, never is: no
    sentence can be referring to that row.  Requiring every component is what
    separates a citable aggregate from a per-instance dump, and it errs toward
    unresolved, which is the direction that gets reported as our own coverage
    loss rather than as the agent's failure.
    """
    if nameable is None:
        return True
    return all(tokens(part) & nameable for part in name.split("|") if part.strip())


def _is_num(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False



def _walk_python(path, out, nameable=None):
    """Index literal numeric settings from released source code.

    D3 face A puts code out of scope as a CLAIM type -- prose and proofs carry
    no artifact-side number. That is not the same as putting code out of scope
    as an ARTIFACT. `hyperparam` is a declared MLRC assertion type ("all
    hyper-parameters used to generate results"), and for an agent that ships its
    script and no config table, the script IS where that value lives. The first
    external arm made the distinction unavoidable: its released summary records
    the code that produced the numbers and not the numbers, so a JSON-only index
    finds nothing to check "batch=32" against.

    Parsed with ast, never regex. A regex over source would match the same token
    in a comment, a string or dead code, and the resulting key would name a
    setting the run never used.

    Two shapes are indexed, both unambiguous:
        name = <number>                  ->  file:name
        call(..., kw=<number>)           ->  file:call.kw
    Defaults of an uncalled signature are NOT indexed: a default is what the
    code would do if the caller said nothing, and the caller usually says
    something.
    """
    import ast
    try:
        tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    except (SyntaxError, ValueError, OSError):
        raise ValueError(f"unparseable python: {path}")

    # A content hash in the filename is not part of a setting's identity, and
    # leaving it in is actively harmful: key_tokens splits it into fragments
    # like "2", "14", "65", which then match numerals in the manuscript and
    # manufacture resolutions. The identity of a setting is its NAME.
    stem = re.sub(r"[_-][0-9a-f]{8,}(?=\.py$)", "", pathlib.Path(path).name)

    def num(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = num(node.operand)
            return None if inner is None else -inner
        return None

    def add(key, value):
        if value is None:
            return
        if _citable(key, nameable):
            out[key] = value

    # A SETTING is something handed to a callee, or declared once at module
    # level. A local initialised to zero inside a loop is an accumulator, and
    # indexing `correct = 0.0` or `total_train_loss = 0.0` invites a claim about
    # accuracy to resolve against a counter's starting value.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            v = num(node.value)
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    add(f"{stem}:{tgt.id}", v)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None) or "call"
            if name in PRESENTATION_CALLS:
                continue
            for kw in node.keywords:
                if kw.arg:
                    add(f"{stem}:{name}.{kw.arg}", num(kw.value))
    return out



def _walk_markdown(path, out, nameable=None):
    """Index Markdown result tables: cell identity is row label + column header.

    This is the same structural reading `_walk_csv` performs, against a
    different serialisation. It is worth having precisely because the first
    external arm failed for want of it: an agent that writes its results as a
    Markdown table has recorded them as DATA -- the row says which condition,
    the header says which quantity -- while one that writes them into a source
    string has not, and the difference decides whether any error rate can be
    attached to it at all.

    Only pipe tables with a delimiter row are read. A pipe character in prose
    does not make a table, and guessing would put prose numerals into the
    artifact index.
    """
    try:
        lines = pathlib.Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        raise ValueError(f"unreadable markdown: {path}")

    stem = pathlib.Path(path).name
    i = 0
    while i < len(lines) - 1:
        row, nxt = lines[i].strip(), lines[i + 1].strip()
        is_delim = (set(nxt) <= set("|-: ") and "-" in nxt and "|" in nxt)
        if not (row.startswith("|") and is_delim):
            i += 1
            continue
        headers = [c.strip() for c in row.strip("|").split("|")]
        i += 2
        while i < len(lines) and lines[i].strip().startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if cells:
                label = cells[0]
                for h, c in zip(headers[1:], cells[1:]):
                    if _is_num(c):
                        key = f"{stem}[{label}|{h}]"
                        if _citable(key.replace("[", "|").replace("]", ""), nameable):
                            out[key] = float(c)
            i += 1
    return out


def artifact_index(run_dir, nameable=None, contract=None, unreadable=None):
    """Every number the run produced that a manuscript could plausibly cite.

    A row is indexed only if the manuscript can NAME it -- if it carries identifier fields
    of its own (which dataset, which model, which metric).  A predictions dump
    has 300k rows keyed by identifiers that appear nowhere in the manuscript,
    so no sentence can be referring to one; an aggregate table is keyed by
    MOOCCubeX and DGEKT and ndcg10, words the results section actually uses.

    `contract` is the run's D3 recording contract (D3-recording-contract.md,
    face B).  Its `reportable_outputs` globs declare which files hold results a
    manuscript may cite.  Without it the tree is unbounded: one real run indexed
    299,345 keys of which ~874 were citable aggregates, and three successive
    automatic cuts each removed the wrong thing (twice deleting the 432-row main
    results table).  The scope is DECLARED, never guessed -- a published manual
    declaration beats a hidden bad heuristic.

    `nameable` is the manuscript's own word set.  Scoping by it depends on the
    manuscript's WORDS, never its values, so the universe stays a property of
    the structure.  Two size-based cuts were tried first and both removed the
    main results table -- size is not what makes a row citable.
    """
    out = {}
    root = pathlib.Path(run_dir)
    # A contract that DECLARES an empty scope is not the same thing as no
    # contract, and the difference must not be decided by truthiness: `[]` is
    # falsy, so an arm whose globs match nothing used to fall through to the
    # unbounded scan -- silently answering "we guessed" to a question the
    # contract had already answered. That is the exact failure D3 exists to
    # prevent, and it would surface later as a small noisy N that section 4.8.7
    # forbids us from reading as the arm's recording discipline.
    globs = (contract or {}).get("reportable_outputs")
    if globs is None:
        files = sorted(root.rglob("*"))           # no contract: unbounded, by admission
    else:
        files = sorted({f for g in globs for f in root.glob(g)})
    for p in files:
        if p.suffix == ".json" and "manuscript" not in p.parts:
            try:
                _walk_json(json.loads(p.read_text()), p.name, out, nameable)
            except (ValueError, OSError) as exc:
                # A file the contract DECLARED and we could not read is our
                # failure, fully attributable, and it used to vanish here. That
                # matters: section 4.8.7 requires every arm to report a parse
                # failure rate separately from its recording discipline, and a
                # silently skipped file is precisely a parse failure being
                # charged to the arm instead of to us.
                if unreadable is not None:
                    unreadable.append({"file": str(p), "error": type(exc).__name__})
                continue
        elif p.suffix in (".md", ".markdown"):
            try:
                _walk_markdown(p, out, nameable)
            except (ValueError, OSError) as exc:
                if unreadable is not None:
                    unreadable.append({"file": str(p), "error": type(exc).__name__})
                continue
        elif p.suffix == ".py":
            try:
                _walk_python(p, out, nameable)
            except (ValueError, OSError) as exc:
                if unreadable is not None:
                    unreadable.append({"file": str(p), "error": type(exc).__name__})
                continue
        elif p.suffix == ".csv":
            try:
                _walk_csv(p, out, nameable)
            except (ValueError, OSError, UnicodeDecodeError) as exc:
                if unreadable is not None:
                    unreadable.append({"file": str(p), "error": type(exc).__name__})
                continue
    return out


# ---------------------------------------------------------------------------
# claim side
# ---------------------------------------------------------------------------

def load_contract(run_dir, contract=None):
    """The run's D3 recording contract: shipped by the agent, or supplied by us.

    An agent that ships none is the common case, not the exception.  We then
    freeze one by hand BEFORE the run is audited, publish it, and report it as
    an experimenter-supplied input -- see D3-recording-contract.md section 2.2.
    """
    if isinstance(contract, dict):
        return contract
    for cand in ([pathlib.Path(contract)] if contract else []) + \
                [pathlib.Path(run_dir) / "record_contract.json"]:
        if cand.exists():
            return json.loads(cand.read_text())
    return None


def manuscript_text(run_dir):
    """The manuscript, assembled the way LaTeX reads it: root + \\input expansion.

    Concatenating the .tex files in filename order is wrong in a way that costs
    most of the universe.  A table's \\caption sits in main.tex at the
    \\input{table_main} site while its tabular sits in table_main.tex, so a
    proximity search for the caption lands in whatever file sorted next -- and
    the caption is often the only place the protocol is named.  Without it a
    results cell ties across every protocol and binds to nothing.

    Files unreachable from the root are excluded: LaTeX would not compile them
    into the document, so the manuscript never made those claims.  A `versions/`
    or `old/` directory of superseded drafts drops out for free.
    """
    root_dir = pathlib.Path(run_dir)
    for pat in ("manuscript", "paper", "."):
        cand = sorted((root_dir / pat).glob("*.tex")) if pat != "." else sorted(root_dir.glob("*.tex"))
        if cand:
            break
    if not cand:
        return "", []

    roots = [f for f in cand if "\\documentclass" in f.read_text(errors="ignore")]
    if not roots:
        return "\n".join(f.read_text(errors="ignore") for f in cand), cand
    used, seen = [], set()

    def expand(path):
        if path in seen or not path.exists():
            return ""
        seen.add(path); used.append(path)
        text = path.read_text(errors="ignore")
        out, last = [], 0
        for m in INPUT_RE.finditer(text):
            out.append(text[last:m.start()])
            child = m.group(1)
            out.append(expand(path.parent / (child if child.endswith(".tex") else child + ".tex")))
            last = m.end()
        out.append(text[last:])
        return "".join(out)

    # EVERY root, not just main.tex: a supplement carries its own
    # \\documentclass and the claims in it are claims the submission makes.
    # Ordered by filename so the assembled text is deterministic.
    return "\n".join(expand(r) for r in roots), used


INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
TABULAR_RE = re.compile(r"\\begin\{tabular\}.*?\\end\{tabular\}", re.S)
CAPTION_RE = re.compile(r"\\caption\{([^}]*)\}")
CELL_SPLIT_RE = re.compile(r"(?<!\\)&")
SPEC_RE = re.compile(r"\\begin\{tabular\}\s*(?:\[[^\]]*\])?\{([^}]*)\}")


def table_cells(text):
    """(start, end, frame) for every cell inside a tabular, sorted by start.

    Most of an agent's numbers are table cells, and a cell's identity is its ROW
    LABEL and COLUMN HEADER -- which sit tens of characters and several newlines
    away, in two different directions.  A flat character window around the
    numeral cannot see either, so a window-only extractor drops the majority of
    the universe and reports the loss as if it were the agent's fault.
    """
    spans = []
    for tbl in TABULAR_RE.finditer(text):
        body = tbl.group(0)
        cap = CAPTION_RE.search(text[max(0, tbl.start() - 400): tbl.start()])
        caption = cap.group(1) if cap else ""

        # Which columns hold labels and which hold numbers is declared in the
        # column spec, and where the header ends is declared by \midrule.  Both
        # are read rather than guessed: "the cell has no digits" gets NDCG@10
        # and Recall@10 wrong, and those are exactly the columns that matter.
        spec = SPEC_RE.search(body)
        letters = re.sub(r"[^lcr]", "", spec.group(1)) if spec else ""
        label_cols = {i for i, ch in enumerate(letters) if ch in "lc"}

        header, offset, seen_midrule = None, tbl.start(), False
        for row in body.split(r"\\"):
            cells, cursor = [], offset
            for cell in CELL_SPLIT_RE.split(row):
                cells.append((cursor, cursor + len(cell), cell))
                cursor += len(cell) + 1
            if r"\midrule" in row:                       # data starts in THIS chunk
                seen_midrule = True
            if not seen_midrule and len(cells) > 1:
                header = [c for _s, _e, c in cells]       # last row above \midrule
            labels = [c for i, (_s, _e, c) in enumerate(cells) if i in label_cols]
            if not labels:
                labels = [c for _s, _e, c in cells if not NUM_RE.search(c)]
            for i, (s, e, _cell) in enumerate(cells):
                col = header[i] if header and i < len(header) else ""
                spans.append((s, e, f"{caption} {' '.join(labels)} {col}"))
            offset += len(row) + 2
    spans.sort()
    return spans


def _frame_for(pos, spans, masked, start, end):
    """Structural frame if the numeral is a table cell, else a character window."""
    lo, hi = 0, len(spans)
    while lo < hi:                                   # rightmost span starting <= pos
        mid = (lo + hi) // 2
        if spans[mid][0] <= pos:
            lo = mid + 1
        else:
            hi = mid
    if lo and spans[lo - 1][1] >= pos:
        s, e, frame = spans[lo - 1]
        return frame, s, e
    return masked[max(0, start - FRAME_CHARS): end + FRAME_CHARS], None, None


def claim_positions(text):
    """(start, end, literal, dp, frame_tokens, assertion_type) for each numeral."""
    masked = PLUMBING_RE.sub(lambda m: " " * len(m.group(0)), text)
    masked = CITATION_YEAR_RE.sub(lambda m: " " * len(m.group(0)), masked)
    spans = table_cells(text)
    out = []
    for m in NUM_RE.finditer(masked):
        lit = m.group(0)
        frame, cs, ce = _frame_for(m.start(), spans, masked, m.start(), m.end())
        toks = tokens(frame)
        atype = assertion_type(toks)
        if atype is None:
            continue
        role = None
        if cs is not None:
            if r"\pm" in text[cs:m.start()]:
                role = "dispersion"
            elif r"\pm" in text[m.end():ce]:
                role = "center"
        rel = RELATION_RE.search(text[max(0, m.start() - 12):m.start()])
        relation = RELATION_MAP.get(rel.group(1)) if rel else None
        dp = len(lit.split(".")[1]) if "." in lit else 0
        out.append((m.start(), m.end(), lit, dp, toks, atype, role, relation))
    return out


def bind(frame_toks, index_tokens, role=None, atype=None, tie_out=None):
    """Resolve a frame to an artifact key by WORDS ONLY.  Ties are unresolved.

    Returns (key, reason).  `reason` is None on success and otherwise names why
    the resolution failed, because the failures do not all belong to the same
    party -- see the comment at the `if not scored` branch.

    Two conditions, both required.  The frame must name the LEAF (the quantity:
    `mean`, `ndcg10`, `accuracy`) and must also share a token with the record's
    identifiers (which dataset, which model).  Naming the quantity without
    saying which one, or the other way round, is not a resolution -- it is an
    ambiguity, and an ambiguity we resolve by guessing would show up later as an
    error rate we cannot explain.  Unresolved is the honest outcome and it is
    reported as OUR coverage loss, never as the agent's.
    """
    # Two key families with different identity semantics, not one family with a
    # tunable threshold. A results-table key is identified by WHICH ROW as well
    # as which quantity -- naming `ndcg10` without naming the dataset is a
    # genuine ambiguity. A code setting has no row: `batch_size` is the whole
    # identity, and no paper writes "the DataLoader's batch size". Requiring a
    # second identifier token there imports the table's semantics into a place
    # it does not hold, and it cost the first external arm nearly every
    # hyper-parameter it prints. Ties stay unresolved either way.
    scored, touched = {}, False
    for key, (leaf, ident) in index_tokens.items():
        # A code setting has no row to name, so it does not get the table's
        # two-token requirement. Scoring stays on (ident + leaf) for both
        # families: dropping ident for code keys was tried and cost the two
        # resolutions this arm has, which is the point at which tuning the rule
        # against one task's N stopped being design and became fitting.
        # CONJUNCTIVE, as this function's contract has always specified: name
        # the quantity AND say which record. The previous implementation summed
        # the two overlaps against a threshold of two, which two leaf tokens
        # satisfy on their own -- so `Throughput (tokens/s)` resolved from the
        # words "throughput tokens" while the manuscript never said which
        # system. The negative control found it: with nothing planted, nearly
        # every bound slot was accused, because nearly every binding was to the
        # wrong row. Requiring both halves is not a tightening of the rule, it
        # is the rule the docstring states.
        hit_ident, hit_leaf = len(ident & frame_toks), len(leaf & frame_toks)
        if CODE_KEY_RE.search(key):
            need, score = 1, hit_ident + hit_leaf   # a setting has no row
        elif hit_ident and hit_leaf:
            need, score = MIN_OVERLAP, hit_ident + hit_leaf
        else:
            need, score = MIN_OVERLAP, 0
        if score:
            touched = True                        # something in the run is ABOUT this
        if score >= need:
            scored.setdefault(score, []).append((key, leaf))
    # A setting claim resolves against settings when the run recorded any.
    if atype in SETTING_TYPES and scored:
        code_only = {s: [(k, lf) for k, lf in ks if CODE_KEY_RE.search(k)]
                     for s, ks in scored.items()}
        code_only = {s: ks for s, ks in code_only.items() if ks}
        if code_only:
            scored = code_only

    if not scored:
        # Nothing recorded even mentions this claim's subject vs. the run does
        # hold related keys and our matcher could not pin one down. Section
        # 4.8.7 says these support OPPOSITE conclusions -- the arm's recording
        # discipline against our own coverage -- so they must not share a
        # counter. The split is mechanical, and it is a decomposition rather
        # than a ground truth: it assumes our tokenizer would have produced SOME
        # overlap had the artifact been recorded at all.
        return None, ("weak_overlap" if touched else "no_candidate")

    top = max(scored)
    best = scored[top]
    if len(best) > 1 and role:                    # break the mean/sd tie by role
        wanted = CENTER_LEAVES if role == "center" else DISPERSION_LEAVES
        best = [(k, lf) for k, lf in best if lf & wanted] or best
    if tie_out is not None and len(best) > 1:
        tie_out.append([k for k, _ in best])
    return (best[0][0], None) if len(best) == 1 else (None, "ambiguous")


def key_tokens(key):
    """Split an artifact key into (leaf tokens, identifier tokens)."""
    leaf = key.rsplit(".", 1)[-1] if "." in key else key
    return tokens(leaf), tokens(key) - tokens(leaf)


def extract(run_dir, contract=None, text=None):
    """The universe U of one run, plus the coverage losses that are OUR fault."""
    src, tex_files = manuscript_text(run_dir)
    text = src if text is None else text
    contract = load_contract(run_dir, contract)
    unreadable = []
    index = artifact_index(run_dir, tokens(text), contract, unreadable)
    index_tokens = {k: key_tokens(k) for k in index}

    slots, unresolved = {}, {"no_candidate": 0, "weak_overlap": 0, "ambiguous": 0}
    for start, end, lit, dp, toks, atype, role, relation in claim_positions(text):
        key, why = bind(toks, index_tokens, role, atype)
        if key is None:
            unresolved[why] += 1
            continue
        name = f"{atype}::{key}::0"
        slot = slots.setdefault(name, {
            "type": atype, "artifact_key": key,
            "artifact_value": index[key], "positions": [],
        })
        slot["positions"].append({"start": start, "end": end, "literal": lit,
                                  "dp": dp, "relation": relation})

    # A slot whose copies disagree is already internally inconsistent.  Planting
    # it would silently REPAIR that inconsistency, so the replicate would differ
    # from the original in a way the planting manifest does not describe.  Drop
    # it from the universe before any draw, deterministically, and report the
    # count -- it is a finding about the manuscript, not a knob.
    bounded = [n for n, s in slots.items()
               if any(p["relation"] for p in s["positions"])]
    for n in bounded:
        del slots[n]

    inconsistent = []
    for name, s in list(slots.items()):
        printed = {round(float(p["literal"]), min(q["dp"] for q in s["positions"]))
                   for p in s["positions"]}
        if len(printed) > 1:
            inconsistent.append(name)
            del slots[name]

    # Two opposite conclusions, kept apart. `no_candidate` says the run holds
    # nothing on this claim's subject -- the arm's recording discipline.
    # `weak_overlap` and `ambiguous` say the run does hold related keys and our
    # matcher could not pin one down -- ours, and the numerator of the parse
    # failure rate section 4.8.7 requires each arm to report separately.
    ours = unresolved["weak_overlap"] + unresolved["ambiguous"]
    total_claims = len(slots) + sum(unresolved.values())
    return {
        "run": str(run_dir), "tex": [p.name for p in tex_files],
        "N": len(slots), "slots": slots,
        "bounded": bounded,                       # asserted as a bound, not a value
        "unresolved_claims": sum(unresolved.values()),
        "unresolved_by_cause": dict(unresolved),
        "unreadable_declared_files": unreadable,  # ours, and fully attributable
        "parse_failure_rate": round(ours / total_claims, 4) if total_claims else 0.0,
        "not_recorded_rate": round(unresolved["no_candidate"] / total_claims, 4)
                             if total_claims else 0.0,
        "internally_inconsistent": inconsistent,  # the manuscript's problem
        "artifact_keys": len(index),
    }


# ---------------------------------------------------------------------------
# planting: claim side only, artifacts byte-identical
# ---------------------------------------------------------------------------

def perturb(value, delta, coarsest_dp, rng):
    """Move a value far enough that even its coarsest printed copy changes.

    Direction follows the pressure a generator under selection actually feels:
    p-values shrink, everything else inflates.
    """
    if value == 0:
        return None
    nv = value * (1 - delta) if 0 < abs(value) < 1e-3 else value * (1 + delta)
    step = 10 ** (-coarsest_dp) if coarsest_dp else 1
    if round(nv, coarsest_dp) == round(value, coarsest_dp):
        nv = value + step * (1 if value >= 0 else -1)
    return nv


def plant(run_dir, k, delta, seed, contract=None, stratified=False):
    """Draw K slots and rewrite every copy of each, coherently.

    `stratified` draws k slots from EACH non-empty stage's own universe instead
    of k from the pooled one, and it is the setting the layered certificate
    requires. Under a pooled draw each stage's K_l is a random variable, and the
    per-stage test adopted from the companion work conditions on a FIXED K --
    so a pooled draw would force this paper to re-derive the within-layer
    machinery it is supposed to be adopting unchanged. Stratifying costs a few
    lines here and keeps the borrowed theorem applicable stage by stage.

    Stratifying does NOT make the per-stage e-values independent: one
    adjudicator sees the whole package, so the stages stay coupled. That is why
    the merge has to be a procedure valid under arbitrary dependence, and the
    two facts must not be used to substitute for one another.
    """
    uni = extract(run_dir, contract)
    text, tex_files = manuscript_text(run_dir)
    names = sorted(uni["slots"])
    # delta is deliberately NOT in the seed: the draw must be the same set of
    # slots at every concealment level, or a delta sweep confounds "the
    # fabrication got smaller" with "we planted somewhere else".
    if stratified:
        drawn = []
        for L, pool in strata(uni["slots"]).items():
            if not pool:
                continue
            # Per-stage stream, so adding or removing a stage cannot shift the
            # draw of the stages beside it.
            r = random.Random(f"{seed}|{run_dir}|{k}|{L}")
            drawn += r.sample(pool, min(k, len(pool))) if k else []
        rng = random.Random(f"{seed}|{run_dir}|{k}|perturb")
    else:
        rng = random.Random(f"{seed}|{run_dir}|{k}")
        drawn = rng.sample(names, min(k, len(names))) if k else []

    edits, ledger = [], []
    for name in drawn:
        s = uni["slots"][name]
        coarsest = min(p["dp"] for p in s["positions"])
        base = float(s["positions"][0]["literal"])
        nv = perturb(base, delta, coarsest, rng)
        if nv is None:
            continue
        for p in s["positions"]:
            rendered = f"{nv:.{p['dp']}f}" if p["dp"] else str(int(round(nv)))
            edits.append((p["start"], p["end"], rendered))
        ledger.append({"slot": name, "type": s["type"], "artifact_key": s["artifact_key"],
                       "from": base, "to": nv, "copies": len(s["positions"])})

    out = text
    for start, end, rendered in sorted(edits, reverse=True):
        out = out[:start] + rendered + out[end:]

    per_layer = {L: [] for L in LAYERS}
    for e in ledger:
        L = TYPE_TO_LAYER.get(e["type"])
        if L:
            per_layer[L].append(e["slot"])

    return out, {"N": uni["N"], "K": len(ledger), "delta": delta, "seed": seed,
                 "stratified": stratified, "universe": names, "planted": ledger,
                 "N_by_layer": {L: len(v) for L, v in strata(uni["slots"]).items()},
                 "K_by_layer": {L: len(v) for L, v in per_layer.items()}}


# ---------------------------------------------------------------------------
# acceptance test: the premise of the certificate, made runnable
# ---------------------------------------------------------------------------

def slot_names_of_text(run_dir, text, contract=None):
    """Slot names of a REWRITTEN manuscript, via the very same extractor.

    This has to be the identical code path, not a re-implementation.  A private
    copy of the walk skipped extract()'s removal of internally-inconsistent
    slots, so the acceptance test compared a filtered set against an unfiltered
    one and reported a universe drift that was purely its own artifact.  On the
    first substrate the filter happened to remove nothing, so the mismatch
    stayed invisible until the universe got big enough to contain one.
    """
    return set(extract(run_dir, contract, text=text)["slots"])


def universe_digest(uni):
    """A stable fingerprint of the universe: the slot NAMES, nothing else.

    Pre-registration is the whole reason Phi has to be deterministic, and until
    now the evidence for it was a hash computed by hand in a shell -- the claim
    lived in a note rather than in the code. Values are deliberately excluded:
    the digest must be identical before and after planting, or it is
    fingerprinting the labelling rather than the universe.
    """
    return hashlib.sha256("\n".join(sorted(uni["slots"])).encode()).hexdigest()


def self_check(run_dir, contract=None):
    uni = extract(run_dir, contract)
    assert uni["N"] > 0, f"empty universe for {run_dir} -- extractor found nothing"

    before = set(uni["slots"])
    planted_text, ledger = plant(run_dir, k=3, delta=0.30, seed=20260825, contract=contract)
    after = slot_names_of_text(run_dir, planted_text, contract)

    # THE premise: planting changes values, never the universe.  If this fails,
    # U is a function of L and the certificate has no argument behind it.
    missing, extra = before - after, after - before
    assert not missing and not extra, (
        f"planting moved the universe\n  lost: {sorted(missing)[:5]}\n  new: {sorted(extra)[:5]}")

    assert ledger["K"] > 0, "nothing was planted, so nothing was tested"
    original, _ = manuscript_text(run_dir)
    assert planted_text != original, "planting changed no text"

    null_text, null_ledger = plant(run_dir, k=0, delta=0.30, seed=20260825, contract=contract)
    assert null_text == original, "K=0 replicate was perturbed"
    assert null_ledger["K"] == 0

    print(f"self-check OK   N={uni['N']}  planted={ledger['K']}  "
          f"unresolved={uni['unresolved_claims']}  "
          f"inconsistent={len(uni['internally_inconsistent'])}")
    print(f"universe digest {universe_digest(uni)}")
    return uni


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--contract", help="D3 recording contract JSON (see D3-recording-contract.md)")
    ap.add_argument("--plant", metavar="RUN")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--delta", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=20260825)
    a = ap.parse_args()

    default = (pathlib.Path("substrates")
               / "substrate-84/blind_reviews/round-10/submission")

    if a.self_check:
        run = pathlib.Path(a.run) if a.run else default
        if not run.exists():
            raise SystemExit(f"no substrate at {run}\nPass --run <submission_dir>.")
        self_check(run, a.contract)
        return

    if a.plant:
        text, ledger = plant(pathlib.Path(a.plant), a.k, a.delta, a.seed, a.contract)
        print(json.dumps({k: v for k, v in ledger.items() if k != "universe"}, indent=1))
        return

    run = pathlib.Path(a.run) if a.run else default
    uni = extract(run, a.contract)
    print(f"N={uni['N']}  artifact_keys={uni['artifact_keys']}  "
          f"unresolved={uni['unresolved_claims']}  "
          f"inconsistent={len(uni['internally_inconsistent'])}")
    by_type = {}
    for s in uni["slots"].values():
        by_type[s["type"]] = by_type.get(s["type"], 0) + 1
    for t, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {t:<12} {n:>4}")
    for name, s in list(uni["slots"].items())[:8]:
        print(f"  {name}  <- {[p['literal'] for p in s['positions']]}")


if __name__ == "__main__":
    sys.exit(main())
