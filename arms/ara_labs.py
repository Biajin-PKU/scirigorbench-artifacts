#!/usr/bin/env python3
"""Third external arm: ARA, and the first one from an independent source.

The first two arms are two slices of one corpus, captured by one group. A
reviewer is entitled to count them as one, and the validity-domain claim rests
on them. ARA is a different release by different authors, and it is the
constructive case rather than another sample of the failure: the format exists
precisely to make a research artifact auditable, so it should satisfy both
conditions the domain requires.

    logic/claims.md          claim side -- an explicit claim listing, which is
                             face C of our recording contract already built
    evidence/tables/*.md     artifact side -- results recorded as data

If the construction fails here it fails on a substrate designed for it, and the
domain is narrower than we have any right to claim.

    python3 arms/ara_labs.py --repo /tmp/ara
"""
import argparse, copy, json, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import phi

CONTRACT = pathlib.Path(__file__).resolve().parent.parent / "contracts/ara-labs.json"

# ARA marks each claim AND says which evidence it rests on. Reading that field
# is the difference between requiring face C of the recording contract and
# CONSUMING it. Without it, a claim sentence naming four numbers across three
# conditions leaves a 90-character window holding several record identifiers at
# once, and every candidate ties.
CLAIM_BLOCK = re.compile(r"^## (C\d+):", re.M)
EVIDENCE_REF = re.compile(r"\*\*Evidence basis\*\*:(.*)$", re.M)
TABLE_REF = re.compile(r"\bTable\s*(\d+)", re.I)


def claim_blocks(claims_md):
    """[(claim id, block text, {table numbers it cites})]"""
    parts, out = CLAIM_BLOCK.split(claims_md), []
    for i in range(1, len(parts), 2):
        cid, body = parts[i], parts[i + 1]
        refs = set()
        for m in EVIDENCE_REF.finditer(body):
            refs |= {int(x) for x in TABLE_REF.findall(m.group(1))}
        out.append((cid, body, refs))
    return out


def scoped_contract(contract, tables):
    """Contract narrowed to the evidence a claim actually cites."""
    c = copy.deepcopy(contract)
    c["reportable_outputs"] = [f"evidence/tables/table{n}_*.md" for n in sorted(tables)]
    return c


def run(ara_dir, contract, use_links=False):
    claims = ara_dir / "logic/claims.md"
    if not claims.exists():
        return None
    text = claims.read_text(encoding="utf-8")

    if use_links:
        # One pass per claim, against only the evidence that claim cites, then
        # union. This is the claim-to-artifact link being used rather than
        # rediscovered from a text window.
        slots, unres, keys = {}, {"no_candidate": 0, "weak_overlap": 0, "ambiguous": 0}, 0
        for cid, body, refs in claim_blocks(text):
            if not refs:
                continue
            u = phi.extract(ara_dir, scoped_contract(contract, refs), text=body)
            keys += u["artifact_keys"]
            for k, v in u["slots"].items():
                slots.setdefault(f"{cid}::{k}", v)
            for k in unres:
                unres[k] += u["unresolved_by_cause"][k]
        total = len(slots) + sum(unres.values())
        ours = unres["weak_overlap"] + unres["ambiguous"]
        uni = {"N": len(slots), "slots": slots, "artifact_keys": keys,
               "unresolved_claims": sum(unres.values()), "unresolved_by_cause": unres,
               "parse_failure_rate": round(ours / total, 4) if total else 0.0,
               "not_recorded_rate": round(unres["no_candidate"] / total, 4) if total else 0.0}
    else:
        uni = phi.extract(ara_dir, contract, text=text)
    st = phi.strata(uni["slots"])
    return {"ara": ara_dir.name, "artifact_keys": uni["artifact_keys"], "N": uni["N"],
            "unresolved": uni["unresolved_claims"], "by_cause": uni["unresolved_by_cause"],
            "parse_failure_rate": uni["parse_failure_rate"],
            "not_recorded_rate": uni["not_recorded_rate"],
            "N_by_layer": {k: len(v) for k, v in st.items() if v},
            "universe_digest": phi.universe_digest(uni)[:16],
            "evaluable": uni["N"] >= 10}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True)
    ap.add_argument("--links", action="store_true",
                    help="consume each claim's Evidence basis pointer")
    ap.add_argument("--out", default="arms/ara_labs_phi.json"); a = ap.parse_args()
    contract = json.loads(CONTRACT.read_text())
    roots = [p for p in (pathlib.Path(a.repo) / "examples").iterdir()
             if p.is_dir() and (p / "logic/claims.md").exists()]
    rows = []
    hdr = f"{'ARA':26s} {'keys':>6} {'N':>5} {'unres':>6} {'ours':>7} {'theirs':>7} {'eval':>5}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(roots):
        row = run(r, contract, use_links=a.links)
        if not row: continue
        rows.append(row)
        print(f"{row['ara'][:26]:26s} {row['artifact_keys']:>6} {row['N']:>5} "
              f"{row['unresolved']:>6} {row['parse_failure_rate']:>6.1%} "
              f"{row['not_recorded_rate']:>6.1%} {'YES' if row['evaluable'] else 'NO':>5}")
    pathlib.Path(a.out).write_text(json.dumps(rows, indent=1))
    if rows:
        print(f"\nevaluable: {sum(r['evaluable'] for r in rows)}/{len(rows)}   "
              f"N max={max(r['N'] for r in rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
