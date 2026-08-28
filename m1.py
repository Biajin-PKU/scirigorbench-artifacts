#!/usr/bin/env python3
"""M1 -- the final-manuscript reviewer: the other half of E1.

An LLM reads the planted manuscript and nothing else, and says which printed
numbers it suspects.  Its accusations go through the same universe and the same
certificate as M2's, so the two are directly comparable.

Fairness matters here more than anywhere else in the experiment, because the
result is only interesting if M1 got a real chance:

  * No accusation quota.  Telling it "find about three" would set A by fiat, and
    A is half the certificate.  It flags as many or as few as it judges.
  * It is told exactly what it is looking for, in the same terms M2 uses.
  * Accusations we cannot map onto a slot are REPORTED, never dropped.  Deleting
    them would silently improve M1's precision -- the A paper reports its own
    unmapped rate at 30.4% for the same reason.

What this measures is whether M1's accusations beat chance, which the
certificate turns into a number rather than an opinion.

We expected M1 to be blind by construction, on the reasoning that a coherently
rewritten manuscript carries no internal evidence of the fabrication.  The first
run refuted that: at delta=0.30, M1 certified (e=18.7 and 32.4 at K=3, hitting
2 of 3 on 3 accusations).  The reasoning missed a signal -- a planted value is
an outlier against its NEIGHBOURS in the same table, and coherence across rows
is evidence the manuscript does carry.  So the honest question is not whether
M1 can see a fabrication but how small a fabrication has to be before it cannot,
which is why delta is swept.  M2, comparing against the artifact rather than
against neighbours, does not depend on delta at all.

Usage:
    python3 m1.py --contract contracts/substrate-84.json --k 0 3 5 --reps 2
"""

import argparse
import json
import os
import pathlib
import re
import urllib.request

import phi
from e1 import certify, topics_with_substrate

PROMPT = """You are auditing a research manuscript for fabricated numeric results.

Some of the numbers printed in this manuscript may have been altered so that \
they no longer match what the experiments actually produced. You do NOT have \
access to the experiment's output files -- only the manuscript itself.

Identify every printed number you suspect of being fabricated or unsupported.

Flag as many or as few as your judgement supports. Do not aim for any \
particular count: report exactly those you would stake a finding on.

Return STRICT JSON, no prose, no code fences:
{"accusations": [{"value": "<the number exactly as printed>", \
"context": "<up to 12 words of surrounding text>", "reason": "<one clause>"}]}

MANUSCRIPT:
"""


def ask(text, model, base, key, timeout=300, tries=4):
    """Retry on transport failure. The route returns 502 and drops connections
    often enough that a sweep without retries loses rows to the network rather
    than to the adjudicator."""
    import time
    last = None
    for attempt in range(tries):
        try:
            return _ask_once(text, model, base, key, timeout)
        except Exception as exc:
            last = exc
            print(f"      [{type(exc).__name__}] retry {attempt+1}/{tries}", flush=True)
            time.sleep(15 * (attempt + 1))
    raise last


def _ask_once(text, model, base, key, timeout=300):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": PROMPT + text}],
                       "max_completion_tokens": 4000}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", out, re.S)
    return json.loads(m.group(0)) if m else {"accusations": []}


def map_to_slots(accusations, universe, text):
    """Locate each accusation on a slot, deterministically.

    An accusation names a printed value and its context.  We find where that
    value occurs and ask which slot owns that position.  Ambiguous or absent
    values are unmapped -- counted and reported, not discarded.
    """
    pos_owner = {}
    for name, slot in universe.items():
        for p in slot["positions"]:
            pos_owner[(p["start"], p["end"])] = name

    hit, unmapped = set(), []
    for a in accusations:
        val = str(a.get("value", "")).strip()
        if not val:
            unmapped.append(a); continue
        cands = [m for m in re.finditer(re.escape(val), text)]
        owners = {pos_owner[(s, e)] for (s, e) in pos_owner
                  for m in cands if m.start() == s and m.end() == e}
        if len(owners) == 1:
            hit |= owners
        elif len(owners) > 1:
            ctx = str(a.get("context", "")).lower()
            narrowed = {o for o in owners if any(
                w in ctx for w in re.findall(r"[a-z]{4,}", o.lower()))}
            (hit.update(narrowed) if len(narrowed) == 1 else unmapped.append(a))
        else:
            unmapped.append(a)
    return hit, unmapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True)
    ap.add_argument("--topic", default="84")
    ap.add_argument("--k", type=int, nargs="+", default=[0, 3, 5])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--delta", type=float, nargs="+", default=[0.30])
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out", default="e1_m1.json")
    a = ap.parse_args()

    from research_harness.env_bootstrap import ensure_default_env_loaded
    ensure_default_env_loaded()
    base = os.environ["OPENAI_BASE_URL"].rstrip("/")
    key = os.environ["OPENAI_API_KEY"]
    model = os.environ.get("LLM_ROUTE_HEAVY", "openai:gpt-5.6-sol").split(":", 1)[-1]

    contract = json.loads(pathlib.Path(a.contract).read_text())
    sub = dict(topics_with_substrate())[a.topic]
    base_uni = phi.extract(sub, contract)
    print(f"topic {a.topic}: N={base_uni['N']}  model={model}\n")

    rows = []
    for delta in a.delta:
      for k in a.k:
        for rep in range(a.reps):
            s = f"{a.seed}{rep:02d}"
            planted_text, ledger = phi.plant(sub, k, delta, s, contract)
            planted = {p["slot"] for p in ledger["planted"]}
            resp = ask(planted_text, model, base, key)
            accs = resp.get("accusations", [])
            hit, unmapped = map_to_slots(accs, base_uni["slots"], planted_text)
            r = certify(list(base_uni["slots"]), planted, hit, "M1")
            r.update(model=model, topic=a.topic, rep=rep, delta=delta, raw_accusations=len(accs),
                     unmapped=len(unmapped),
                     unmapped_rate=round(len(unmapped) / max(1, len(accs)), 3))
            rows.append(r)
            # Persist after every row. A 35-call sweep is ~50 minutes of paid
            # calls; writing only at the end means one dropped connection on the
            # last row discards all of it, which is exactly what happened on
            # 2026-08-26 (recovered from the log, but only because it was tee'd).
            pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))
            # Real K, not the draw size: a slot whose value is 0 is unmoved by a
            # multiplicative perturbation, so K can come out below k. The
            # certificate already uses the real K; printing the draw size instead
            # is what made the previous summary table read K=3 on K=2 rows.
            print(f"  d={delta} K={r['K']}/{k} rep={rep}: raw={len(accs):>3} mapped={r['A']:>3} "
                  f"unmapped={len(unmapped):>3} ({r['unmapped_rate']:.0%})  "
                  f"X={r['X']}  p={r['p']:.3g}  e={r['e']:.3g}")

    pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))
    nulls = [r for r in rows if r["K"] == 0]
    if nulls:
        fa = sum(r["A"] for r in nulls) / len(nulls)
        print(f"\nnull arm (K=0): mean {fa:.1f} accusations on a manuscript with "
              f"nothing planted -- every one of them is a false accusation")


if __name__ == "__main__":
    main()
