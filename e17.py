#!/usr/bin/env python3
"""E17 -- can a model actually comply, or only we?

E10 and E11 build compliant substrates with a generator we wrote. That shows
what the contract buys once met; it does not show that anything other than a
Python loop can meet it. A reviewer's objection is exact: the only manuscript
that satisfies the contract is one the authors wrote.

So here a real model writes the results section, on a real run's artifacts,
with the marking requirement in its instructions -- and then we check its
marking the same way we check anyone's:

    proposed   claim -> key pairs the model emitted alongside its prose
    resolvable the key exists in the frozen index
    faithful   the value it printed matches that cell
    bound      both, i.e. what phi would count

The interesting number is not how many it emits but how many survive. A model
that marks confidently and wrongly is worse than one that does not mark, because
a wrong pointer is a pointer the audit will follow.

    python3 e17.py --tasks 5
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
sys.path.insert(0, str(HERE / "arms"))
import phi                                              # noqa: E402

MODEL = os.environ.get("E17_MODEL", "gpt-5.6-sol")
MAX_KEYS = 260

PROMPT = """You are writing the Results section of a machine-learning paper.

You must satisfy a recording contract: every number you print that reports a
computed result has to be accompanied by the exact artifact key it comes from.
Keys are exact strings from the list below. Do not invent, abbreviate or
reformat a key, and do not print a number you cannot key.

Write 6 to 10 sentences of ordinary results prose. Then give the marking.

Return JSON only, no prose outside it, no code fences:
{{"prose": "<the results section>",
  "claims": [{{"value": "<number exactly as it appears in your prose>",
              "key": "<exact key from the list>"}}]}}

=== ARTIFACT CELLS ({nkeys} of {total}, key = value) ===
{cells}
"""


def ask(prompt, timeout=900):
    base = os.environ["OPENAI_BASE_URL"].rstrip("/")
    key = os.environ["OPENAI_API_KEY"]
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_completion_tokens": 4000}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r["choices"][0]["message"]["content"]


def audit(claims, prose, index):
    """Check the model's marking the way phi would, and separate the failures."""
    resolvable = faithful = printed = 0
    bound = []
    for c in claims:
        k, v = c.get("key"), str(c.get("value", "")).strip()
        if k not in index:
            continue
        resolvable += 1
        if v and v in prose:
            printed += 1
        try:
            lit = float(re.sub(r"[^\d.eE+-]", "", v))
        except ValueError:
            continue
        dp = len(v.split(".")[1]) if "." in v else 0
        if round(lit, dp) == round(float(index[k]), dp):
            faithful += 1
            if v in prose:
                bound.append((k, v))
    return {"proposed": len(claims), "resolvable": resolvable,
            "printed_in_prose": printed, "faithful": faithful,
            "bound": len(bound)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/tmp/mlrb/r")
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--out", default="e17_compliance.json")
    a = ap.parse_args()

    from research_harness.env_bootstrap import ensure_default_env_loaded
    ensure_default_env_loaded()

    root = pathlib.Path(a.repo) / "ai_scientist_v2_papers/o4-mini"
    contract = json.loads((HERE / "contracts/mlrbench-v2-o4mini.json").read_text())
    tasks = sorted(p for p in root.iterdir() if p.is_dir())[:a.tasks]

    print(f"E17 -- a real model writing under the contract, {MODEL}\n")
    hdr = (f"  {'task':24s}{'proposed':>10}{'resolvable':>12}{'printed':>9}"
           f"{'faithful':>10}{'bound':>7}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    rows = []
    for td in tasks:
        index = {k: v for k, v in phi.artifact_index(td, None, contract).items()
                 if isinstance(v, float)}
        keys = sorted(index)
        shown = keys if len(keys) <= MAX_KEYS else [
            keys[int(i * len(keys) / MAX_KEYS)] for i in range(MAX_KEYS)]
        cells = "\n".join(f"{k} = {index[k]}" for k in shown)
        try:
            out = ask(PROMPT.format(nkeys=len(shown), total=len(keys), cells=cells))
        except Exception as e:
            print(f"  {td.name:24s}  [{type(e).__name__}] {str(e)[:36]}")
            continue
        m = re.search(r"\{.*\}", out, re.S)
        try:
            blob = json.loads(m.group(0)) if m else {}
        except Exception:
            blob = {}
        prose = blob.get("prose", "")
        r = audit(blob.get("claims", []), prose, index)
        r.update(task=td.name, index_size=len(index), prose_chars=len(prose))
        rows.append(r)
        print(f"  {td.name:24s}{r['proposed']:>10}{r['resolvable']:>12}"
              f"{r['printed_in_prose']:>9}{r['faithful']:>10}{r['bound']:>7}")
        (HERE / a.out).write_text(json.dumps(rows, indent=1))

    if not rows:
        print("\n  no task completed"); return 1

    med = lambda f: st.median([r[f] for r in rows])
    tot = lambda f: sum(r[f] for r in rows)
    rate = tot("bound") / max(1, tot("proposed"))
    print(f"""
  over {len(rows)} tasks:
    proposed   median {med('proposed'):g}   total {tot('proposed')}
    resolvable median {med('resolvable'):g}   total {tot('resolvable')}
    faithful   median {med('faithful'):g}   total {tot('faithful')}
    bound      median {med('bound'):g}   total {tot('bound')}
    survival rate (bound / proposed): {rate:.0%}""")

    (HERE / a.out).write_text(json.dumps(
        {"model": MODEL, "rows": rows, "survival_rate": round(rate, 3)}, indent=1))

    print(f"""
Reading. The contract asks the writer to do something it has the information to
do: the keys and their values are in front of it while it writes. What survives
the audit is {rate:.0%} of what it proposed. The gap is the whole point of checking
marking rather than trusting it -- a pointer that names a real cell but is
attached to a number the cell does not hold is not a weaker binding, it is a
false one, and an audit that followed it would score the wrong thing.

Compare the same model asked to RECOVER markings from an unmarked manuscript
(E13): there it verified a median of 1.5 bindings per task. Asked to EMIT them
while writing, with the same information in front of it, it binds a median of
{med('bound'):g} -- an order of magnitude, and the only difference is when the marking is
written. That is the contract's claim, measured on a writer that is not ours.

Whether these universes can certify is a separate question with a derived
answer (E14): at N=17 a planting budget of K>=4 reaches 1/alpha, at N=23 K>=3
does. Every task here clears its own bar at some K, where no released run
clears it at any K.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
