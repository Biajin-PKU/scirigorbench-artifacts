#!/usr/bin/env python3
"""E22 -- what shape is the coverage loss, exactly.

The paper attributes every unresolved claim to one of two opposite causes: the
run recorded nothing on the subject (the arm's), or our matcher could not narrow
the candidates to one (ours). That split is reported in the body. What was not
reported from any artifact is the SHAPE of the second cause, and an earlier
revision printed a breakdown whose parts did not sum to its whole.

The distinction that matters is whether an ambiguous claim ties across several
different records or across statistics of one record. The first is a naming
problem -- the manuscript never said which record it meant -- and the recording
contract addresses it. The second would be a precision problem in our matcher,
and no contract fixes that. So the number decides whether the paper's own
prescription is responsive to its own measurement.

    python3 e22.py
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phi                                                    # noqa: E402
from e1 import topics_with_substrate                          # noqa: E402


def record_of(key):
    """Everything before the final dot-separated leaf, i.e. which record."""
    return key.rsplit(".", 1)[0] if "." in key else key


def main():
    contract = json.loads(
        (HERE / "contracts/substrate-84.json").read_text())
    run = dict(topics_with_substrate())["84"]

    full = phi.extract(run, contract)
    by_cause = full["unresolved_by_cause"]
    ours = by_cause["weak_overlap"] + by_cause["ambiguous"]
    total_claims = full["N"] + sum(by_cause.values())

    # Re-run the binder over the same claims, capturing the tied candidate sets
    # that `bind` discards. tie_out is an optional out-parameter and does not
    # change what bind returns, so this observes the frozen matcher rather than
    # a variant of it.
    src, _ = phi.manuscript_text(run)
    idx = phi.artifact_index(run, phi.tokens(src), phi.load_contract(run, contract), [])
    itok = {k: phi.key_tokens(k) for k in idx}

    ties = []
    for _s, _e, _lit, _dp, toks, atype, role, _rel in phi.claim_positions(src):
        out = []
        phi.bind(toks, itok, role, atype, tie_out=out)
        if out:
            ties.append(out[0])

    within = sum(1 for t in ties if len({record_of(k) for k in t}) == 1)
    cross = len(ties) - within
    sizes = {}
    for t in ties:
        sizes[len(t)] = sizes.get(len(t), 0) + 1

    print(f"""E22 -- coverage attribution on the development substrate

  bound slots (N)                    {full['N']}
  unresolved claims                  {sum(by_cause.values())}
    the run recorded nothing         {by_cause['no_candidate']}
    ours: candidates, none narrowed  {ours}
  total claims                       {total_claims}
  parse failure   {ours}/{total_claims} = {ours/total_claims:.4f}
  not recorded    {by_cause['no_candidate']}/{total_claims} = {by_cause['no_candidate']/total_claims:.4f}

  of the {len(ties)} ambiguous claims, every one is a tie between candidate keys:
    tie spans SEVERAL records        {cross}  ({cross/len(ties):.1%})
    tie inside ONE record            {within}  ({within/len(ties):.1%})
    tie sizes                        {dict(sorted(sizes.items()))}""")

    assert cross + within == len(ties), "the parts must sum to the whole"
    assert ours == len(ties) or by_cause["weak_overlap"], \
        "every ours-claim should have a recorded tie"

    out = {"N": full["N"], "unresolved": sum(by_cause.values()),
           "by_cause": by_cause, "ours": ours, "total_claims": total_claims,
           "parse_failure_rate": round(ours / total_claims, 4),
           "not_recorded_rate": round(by_cause["no_candidate"] / total_claims, 4),
           "ambiguous_with_tie": len(ties), "cross_record": cross,
           "within_record": within,
           "cross_record_share": round(cross / len(ties), 4),
           "tie_sizes": {str(k): v for k, v in sorted(sizes.items())}}
    (HERE / "e22_ambiguity.json").write_text(json.dumps(out, indent=1))

    print(f"""
Reading. {cross/len(ties):.1%} of what our matcher cannot resolve is a tie between
DIFFERENT records, not between statistics of one. The manuscript named a
quantity and never said which run it came from, and no amount of matcher
precision recovers a record the text does not name. That is the failure the
recording contract is written against, which is the check this experiment
exists to perform: a prescription aimed at the minority cause would be a
prescription this paper's own data does not support.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
