#!/usr/bin/env python3
"""E13 -- a stronger extractor as a ceiling, not as a pipeline.

Our matcher is a frame-overlap heuristic with published thresholds. The paper's
central negative result is that no released run records enough to be scored, and
a reviewer is entitled to ask whether that is a fact about the runs or about the
matcher. The honest way to bound it is to hand the same inputs to something
much stronger and see how much further it gets.

    what the LLM sees   the manuscript text, and the list of artifact keys the
                        frozen contract admits -- exactly what our matcher sees
    what it returns     claim -> artifact key bindings, at cell granularity
    what counts         only bindings whose key EXISTS in the index and whose
                        cited value MATCHES that cell to the printed precision

That last line is the whole experiment. An LLM asked to bind claims will invent
keys, and a raw count of what it proposes is not an upper bound on anything. We
therefore verify every binding against the frozen index and report proposed and
verified separately.

This is a CEILING ESTIMATE. It is deliberately outside the pre-registered
pipeline: it is not deterministic, not reproducible from a seed, and cannot be
frozen, so it can bound the negative result but can never produce a certificate.

    python3 e13.py --repo /tmp/mlrb/r --tasks 5
"""

import argparse
import json
import os
import pathlib
import re
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "arms"))
import phi                                              # noqa: E402
from arms.mlrbench_v2 import manuscript_text            # noqa: E402

MODEL = os.environ.get("E13_MODEL", "gpt-5.6-sol")
MAX_KEYS = 1200         # the full index for every task in this corpus; an
                        # earlier run capped at 400 and withheld two thirds of
                        # the keys from the three largest tasks, which is the
                        # handicap this experiment exists to avoid


def ask(prompt, timeout=900, tries=4):
    import time
    last = None
    for a in range(tries):
        try:
            return _ask_once(prompt, timeout)
        except Exception as exc:
            last = exc
            print(f"      [{type(exc).__name__}] retry {a+1}/{tries}", flush=True)
            time.sleep(15 * (a + 1))
    raise last


def _ask_once(prompt, timeout=900):
    base = os.environ["OPENAI_BASE_URL"]
    key = os.environ["OPENAI_API_KEY"]
    body = json.dumps({"model": MODEL,
                       "messages": [{"role": "user", "content": prompt}],
                       "max_completion_tokens": 4000}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r["choices"][0]["message"]["content"]


PROMPT = """You are extracting auditable claims from a research manuscript.

A CLAIM is a number printed in the manuscript that reports a value computed by
the run. For each one, say which artifact cell it reports, choosing from the key
list below. Keys are exact strings; do not invent, abbreviate or reformat them.

Return JSON only, no prose:
{{"bindings": [{{"value": "<the numeral as printed>", "key": "<exact key>"}}]}}

Omit any claim you cannot bind to a key in the list. Precision is worth more
than recall here: a wrong binding is worse than a missing one.

=== ARTIFACT KEYS ({nkeys} shown{trunc}) ===
{keys}

=== MANUSCRIPT ===
{text}
"""


def verify(bindings, index):
    """A binding counts only if the key is real and the value matches the cell."""
    ok, bad_key, bad_value = [], 0, 0
    for b in bindings:
        k, v = b.get("key"), str(b.get("value", "")).strip()
        if k not in index:
            bad_key += 1
            continue
        try:
            lit = float(re.sub(r"[^\d.eE+-]", "", v))
        except ValueError:
            bad_value += 1
            continue
        cell = float(index[k])
        dp = len(v.split(".")[1]) if "." in v else 0
        if round(lit, dp) == round(cell, dp):
            ok.append((k, v))
        else:
            bad_value += 1
    return ok, bad_key, bad_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tasks", type=int, default=5)
    a = ap.parse_args()

    from research_harness.env_bootstrap import ensure_default_env_loaded
    ensure_default_env_loaded()

    root = pathlib.Path(a.repo) / "ai_scientist_v2_papers/o4-mini"
    contract = json.loads((HERE / "contracts/mlrbench-v2-o4mini.json").read_text())
    ours = {r["task"]: r for r in json.loads(
        (HERE / "arms/mlrbench_v2_phi.json").read_text())}

    tasks = sorted(p for p in root.iterdir() if p.is_dir())[:a.tasks]
    rows = []
    print(f"E13 -- ceiling estimate with {MODEL}\n")
    hdr = f"{'task':24s}{'ours':>6}{'proposed':>10}{'verified':>10}{'bad key':>9}{'bad val':>9}"
    print(hdr); print("-" * len(hdr))

    for td in tasks:
        text = manuscript_text(td)
        index = phi.artifact_index(td, None, contract)
        numeric = {k: v for k, v in index.items() if isinstance(v, float)}
        keys = sorted(numeric)
        trunc = ""
        if len(keys) > MAX_KEYS:
            step = len(keys) / MAX_KEYS
            keys = [keys[int(i * step)] for i in range(MAX_KEYS)]
            trunc = f" of {len(numeric)}, evenly sampled"
        prompt = PROMPT.format(nkeys=len(keys), trunc=trunc,
                               keys="\n".join(keys), text=text[:24000])
        try:
            out = ask(prompt)
        except Exception as e:
            print(f"{td.name:24s}  [{type(e).__name__}] {str(e)[:40]}")
            continue
        m = re.search(r"\{.*\}", out, re.S)
        try:
            bindings = json.loads(m.group(0))["bindings"] if m else []
        except Exception:
            bindings = []
        good, bad_key, bad_val = verify(bindings, numeric)
        n_ours = ours.get(td.name, {}).get("N", 0)
        rows.append({"task": td.name, "ours": n_ours, "proposed": len(bindings),
                     "verified": len(good), "bad_key": bad_key,
                     "bad_value": bad_val, "index_size": len(numeric),
                     "keys_shown": len(keys)})
        print(f"{td.name:24s}{n_ours:6d}{len(bindings):10d}{len(good):10d}"
              f"{bad_key:9d}{bad_val:9d}")

    if not rows:
        print("\nno task completed"); return 1
    import statistics as st
    mo = st.median([r["ours"] for r in rows])
    mv = st.median([r["verified"] for r in rows])
    mp = st.median([r["proposed"] for r in rows])
    (HERE / "e13_ceiling.json").write_text(json.dumps(
        {"model": MODEL, "rows": rows,
         "median_ours": mo, "median_verified": mv, "median_proposed": mp}, indent=1))

    print(f"""
  median N, our matcher      : {mo:g}
  median bindings proposed   : {mp:g}
  median bindings verified   : {mv:g}

Reading. The gap between proposed and verified is the reason a raw LLM count is
not an upper bound: a binding that names a key the index does not contain, or
cites a value the cell does not hold, is not evidence that the information was
there to find. What bounds our negative result is the verified column.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
