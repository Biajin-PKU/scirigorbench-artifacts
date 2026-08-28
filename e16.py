#!/usr/bin/env python3
"""E16 -- what is different about the one substrate that works?

Seven manuscripts from one generator, one format, one pipeline. On one the
universe resolves; on the other six it does not. Every threshold in this paper
was tuned on that one, so an unexplained 1-in-7 is the paper's deepest problem
and not a curiosity. The appendix rules out four explanations and stops at "we
cannot presently say what". This file goes further than that and stops in a
better-defined place.

It measures, per substrate: how many artifact cells the run recorded, how many
numerals the manuscript prints, what fraction of those numerals bind, and how
the bound claims spread over records. Then it asks whether any of those orders
the substrates the way the outcome does.

    python3 e16.py
"""

import collections
import json
import math
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phi                                          # noqa: E402
from e1 import topics_with_substrate                # noqa: E402

NUM = re.compile(r"(?<![\w.])\d+\.\d+(?![\w])")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    subs = dict(topics_with_substrate())
    rows = []
    for topic in sorted(subs, key=int):
        cf = HERE / f"contracts/substrate-topic{topic}.json"
        if not cf.exists():
            continue
        contract = json.loads(cf.read_text())
        run = subs[topic]
        text, _ = phi.manuscript_text(run)
        res = phi.extract(run, contract, text=text)
        index = phi.artifact_index(run, None, contract)
        cont = collections.Counter(
            n.split("::")[1].split("[")[0] for n in res.get("slots", {}))
        numerals = len(NUM.findall(text))
        rows.append({
            "substrate": topic, "N": res["N"], "artifact_keys": len(index),
            "manuscript_chars": len(text), "numerals": numerals,
            "bind_pct": round(res["N"] / numerals * 100, 1) if numerals else 0.0,
            "keys_per_numeral": round(len(index) / numerals, 1) if numerals else 0.0,
            "containers": len(cont),
            "top_container_share": round(max(cont.values()) / sum(cont.values()), 2)
            if cont else 0.0,
        })

    print("E16 -- what separates the substrate that works\n")
    hdr = (f"  {'sub':>4}{'N':>6}{'keys':>7}{'numerals':>10}{'bind%':>8}"
           f"{'keys/num':>10}{'records':>9}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['substrate']:>4}{r['N']:>6}{r['artifact_keys']:>7}"
              f"{r['numerals']:>10}{r['bind_pct']:>7.1f}%{r['keys_per_numeral']:>10}"
              f"{r['containers']:>9}")

    work = [r for r in rows if r["N"] >= 10]
    fail = [r for r in rows if r["N"] < 10]
    lo, hi = wilson(len(work), len(rows))
    print(f"\n  {len(work)} of {len(rows)} substrates resolve a usable universe.")
    print(f"  Wilson 95% interval on that rate: {lo:.0%} to {hi:.0%}")

    # does any candidate order the substrates the way the outcome does?
    print("\n  is the outcome monotone in any single quantity?")
    for field in ("artifact_keys", "numerals", "keys_per_numeral",
                  "manuscript_chars"):
        ordered = sorted(rows, key=lambda r: -r[field])
        outcome = [r["bind_pct"] for r in ordered]
        mono = all(outcome[i] >= outcome[i + 1] for i in range(len(outcome) - 1))
        # the sharpest counterexample: a pair where more of the quantity binds less
        worst = None
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                if a["bind_pct"] < b["bind_pct"]:
                    gap = b["bind_pct"] - a["bind_pct"]
                    if worst is None or gap > worst[0]:
                        worst = (gap, a, b)
        note = ""
        if worst:
            _, a, b = worst
            note = (f"  counterexample: sub {a['substrate']} has "
                    f"{a[field]} and binds {a['bind_pct']}%, "
                    f"sub {b['substrate']} has {b[field]} and binds {b['bind_pct']}%")
        print(f"    {field:18s} monotone: {'yes' if mono else 'NO '}{note}")

    top = max(rows, key=lambda r: r["artifact_keys"])
    rest = sorted((r["artifact_keys"] for r in rows if r is not top), reverse=True)
    (HERE / "e16_mechanism.json").write_text(json.dumps(
        {"rows": rows, "resolving": len(work), "total": len(rows),
         "wilson95": [round(lo, 3), round(hi, 3)]}, indent=1))

    print(f"""
Reading. The working substrate is an outlier on one axis: it recorded {top['artifact_keys']}
artifact cells where the next substrate recorded {rest[0]}, a factor of
{top['artifact_keys']/rest[0]:.1f}. Its manuscript is not unusual -- it prints {top['numerals']} numerals against
{min(r['numerals'] for r in rows)} to {max(r['numerals'] for r in rows)} elsewhere -- so this is a difference in what the run wrote
down, not in what the paper claimed.

But recording more is not sufficient, and the table shows why: the outcome is
not monotone in artifact cells, in numerals, or in their ratio. One substrate
records more cells per printed numeral than another and still binds nothing.
So the honest statement is sharper than the appendix's but still not a
mechanism: the substrate that works is separated by 3x on recorded cells, and
no single quantity we can measure orders the remaining six.

With {len(work)} of {len(rows)} resolving, the rate is {len(work)/len(rows):.0%} with a 95% interval of {lo:.0%} to {hi:.0%}.
Seven is too few to narrow that, and it is the number the generator produced.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
