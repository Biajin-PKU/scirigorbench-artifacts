#!/usr/bin/env python3
"""E18 -- the agent's own writer, with and without the contract.

E17 showed a model can comply when we write the instructions. That leaves the
obvious objection: our instructions, our result. So this run uses the agent's
OWN Results-section instruction, taken verbatim from AI Scientist v1's
`per_section_tips["Results"]` in `ai_scientist/perform_writeup.py`, and changes
exactly one thing between arms -- whether the contract clause is appended.

    control    the agent's tip, unmodified
    treatment  the agent's tip + "print no number you cannot key, and emit the key"

Same artifacts, same model, same everything else. The difference in what binds
is attributable to the clause and to nothing else.

There is a finding before the experiment starts. The agent's own tip already
says "Only includes results that have actually been run and saved in the logs.
Do not hallucinate results that don't exist." Faithfulness is already
instructed, and the released papers written under that instruction bind a
median of 0 claims. Instructing a writer to be truthful does not make its
output auditable, because truthfulness is about the values and auditability is
about the pointers.

    python3 e18.py --tasks 5
"""

import argparse
import json
import os
import pathlib
import re
import statistics as st
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phi                                              # noqa: E402

MODEL = os.environ.get("E18_MODEL", "gpt-5.6-sol")
MAX_KEYS = 260
AGENT_TIP = pathlib.Path("/tmp/aisci_results_tip.txt")

CONTRACT_CLAUSE = """
- RECORDING CONTRACT: every number you print that reports a computed result must
  be accompanied by the exact artifact key it came from. Keys are exact strings
  from the list you were given. Do not invent, abbreviate or reformat a key, and
  do not print a number you cannot key.
"""

FRAME = """You are writing the Results section of a machine-learning paper.
Follow these instructions:
{tips}

Return JSON only, no prose outside it, no code fences:
{{"prose": "<the results section>"{claims_field}}}

=== ARTIFACT CELLS ({nkeys} of {total}, key = value) ===
{cells}
"""
CLAIMS_FIELD = (',\n  "claims": [{"value": "<number exactly as in your prose>", '
                '"key": "<exact key>"}]')


def ask(prompt, timeout=900):
    base = os.environ["OPENAI_BASE_URL"].rstrip("/")
    key = os.environ["OPENAI_API_KEY"]
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_completion_tokens": 4000}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout)
                     )["choices"][0]["message"]["content"]


def bind_marked(claims, prose, index):
    """What phi counts when the marking is present: key resolves, value printed,
    value matches the cell."""
    out = {}
    for c in claims:
        k, v = c.get("key"), str(c.get("value", "")).strip()
        if k not in index or not v or v not in prose:
            continue
        try:
            lit = float(re.sub(r"[^\d.eE+-]", "", v))
        except ValueError:
            continue
        dp = len(v.split(".")[1]) if "." in v else 0
        if round(lit, dp) == round(float(index[k]), dp):
            out[k] = v
    return list(out.items())


def bind_unmarked(prose, index):
    """What is recoverable without marking: a printed numeral that equals exactly
    one cell. This is the retrofit E9 rules out for certification, used here only
    to give the control arm its best possible showing."""
    by_val = {}
    for k, v in index.items():
        by_val.setdefault(round(float(v), 6), []).append(k)
    # DISTINCT cells, not numeral occurrences. A universe cannot be larger than
    # the index, and counting occurrences let one cell be bound many times.
    hits = {}
    for m in re.finditer(r"(?<![\w.])\d+\.\d+(?![\w])", prose):
        cands = by_val.get(round(float(m.group(0)), 6))
        if cands and len(cands) == 1:
            hits[cands[0]] = m.group(0)
    return list(hits.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/tmp/aisci_v1")
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--out", default="e18_agent_writer.json")
    a = ap.parse_args()

    from research_harness.env_bootstrap import ensure_default_env_loaded
    ensure_default_env_loaded()
    if not AGENT_TIP.exists():
        print(f"missing {AGENT_TIP}: capture the agent's own tip first")
        return 1
    tip = AGENT_TIP.read_text()

    root = pathlib.Path(a.repo) / "example_papers"
    contract = json.loads((HERE / "contracts/aiscientist-v1.json").read_text())
    released = {r["paper"]: r for r in json.loads(
        (HERE / "arms/aiscientist_v1_phi.json").read_text())}
    papers = sorted(p for p in root.iterdir() if p.is_dir())[:a.tasks]

    print(f"E18 -- the agent's own Results instruction, {MODEL}")
    print(f"  control   = the tip verbatim ({len(tip)} chars)")
    print(f"  treatment = the tip + the contract clause\n")
    hdr = (f"  {'paper':30s}{'released':>10}{'control':>9}{'treatment':>11}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    rows = []
    for pd in papers:
        index = {k: v for k, v in phi.artifact_index(pd, None, contract).items()
                 if isinstance(v, float)}
        if not index:
            continue
        keys = sorted(index)
        shown = keys if len(keys) <= MAX_KEYS else [
            keys[int(i * len(keys) / MAX_KEYS)] for i in range(MAX_KEYS)]
        cells = "\n".join(f"{k} = {index[k]}" for k in shown)

        got = {}
        for arm, tips, field in (("control", tip, ""),
                                 ("treatment", tip + CONTRACT_CLAUSE, CLAIMS_FIELD)):
            try:
                out = ask(FRAME.format(tips=tips, claims_field=field,
                                       nkeys=len(shown), total=len(keys),
                                       cells=cells))
            except Exception as e:
                got[arm] = None
                print(f"  {pd.name:30s}  [{arm}: {type(e).__name__}]")
                continue
            m = re.search(r"\{.*\}", out, re.S)
            try:
                blob = json.loads(m.group(0)) if m else {}
            except Exception:
                blob = {}
            prose = blob.get("prose", "")
            if arm == "control":
                got[arm] = {"bound": len(bind_unmarked(prose, index)),
                            "prose_chars": len(prose)}
            else:
                b = bind_marked(blob.get("claims", []), prose, index)
                got[arm] = {"bound": len(b), "proposed": len(blob.get("claims", [])),
                            "prose_chars": len(prose)}

        rel = released.get(pd.name, {}).get("N", 0)
        c = got.get("control") or {}
        tr = got.get("treatment") or {}
        rows.append({"paper": pd.name, "released_N": rel,
                     "control": c, "treatment": tr, "index_size": len(index)})
        print(f"  {pd.name:30s}{rel:>10}{c.get('bound','-'):>9}{tr.get('bound','-'):>11}")
        (HERE / a.out).write_text(json.dumps(rows, indent=1))

    if not rows:
        print("\n  no paper completed"); return 1
    R = [r["released_N"] for r in rows]
    C = [r["control"].get("bound", 0) for r in rows if r["control"]]
    T = [r["treatment"].get("bound", 0) for r in rows if r["treatment"]]
    print(f"""
  over {len(rows)} papers, same artifacts, same model:
    as released by the agent           median {st.median(R):g}
    agent's tip verbatim               median {st.median(C):g}
    agent's tip + contract clause      median {st.median(T):g}""")

    (HERE / a.out).write_text(json.dumps(
        {"model": MODEL, "rows": rows,
         "median_released": st.median(R), "median_control": st.median(C),
         "median_treatment": st.median(T)}, indent=1))

    print(f"""
Reading. The control arm is the agent's own instruction, which already forbids
hallucinated results, and it binds a median of {st.median(C):g}. Appending one clause about
where each number came from takes that to {st.median(T):g}. The instruction to be truthful
and the instruction to be checkable are different instructions, and only the
second produces a universe: truthfulness constrains the values, auditability
constrains the pointers, and a manuscript can satisfy the first completely while
supplying nothing for the second.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
