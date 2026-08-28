#!/usr/bin/env python3
"""Freeze the instrument before the first external arm runs.

E8 preregistration section 6 lists what has to be fixed before any held-out arm
is touched -- assertion catalogue, matcher, thresholds, task list, the
preregistration itself -- and says so in the sharpest available terms: changing
any of them afterwards turns held-out into in-sample. This script is that step,
made runnable so the claim has an artifact behind it instead of a promise.

It emits a manifest of SHA-256 digests plus the universe digest of the
development substrate. Re-running it later and diffing the manifest answers
"did the instrument move?" mechanically.

    python3 freeze.py --emit          # write FREEZE.json
    python3 freeze.py --verify        # compare against the committed FREEZE.json
"""

import argparse
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent

# Everything whose change would alter what the benchmark measures. Adding to
# this list is cheap; discovering afterwards that something load-bearing was
# outside it is not.
FROZEN_FILES = [
    "phi.py",                       # extractor, planter, catalogue, layer map
    "e1.py",                        # certificate, verbatim from the A paper
    "e2.py",                        # known-ordering recovery
    "m1.py",                        # manuscript-only reference auditor
    "D3-recording-contract.md",     # face A catalogue prose + face B protocol
    "contracts/substrate-84.json",
]

# Thresholds live in code, but naming them here makes a silent edit visible in
# the diff even if the file digest is not inspected by eye.
FROZEN_THRESHOLDS = {
    "MIN_OVERLAP": None,            # filled from phi at emit time
    "FRAME_CHARS": None,
    "KAPPA": None,                  # certificate calibration exponent
    "evaluability_floor_N": 10,     # D3 section 3: below this, not scoreable
}


def sha256_file(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def build():
    sys.path.insert(0, str(HERE))
    import phi
    import e1

    thresholds = dict(FROZEN_THRESHOLDS)
    thresholds["MIN_OVERLAP"] = phi.MIN_OVERLAP
    thresholds["FRAME_CHARS"] = phi.FRAME_CHARS
    thresholds["KAPPA"] = e1.KAPPA

    contract = json.loads((HERE / "contracts/substrate-84.json").read_text())
    sub = dict(e1.topics_with_substrate()).get("84")
    uni = phi.extract(sub, contract) if sub else None

    return {
        "schema": "scirigorbench.freeze.v1",
        "why": "E8 preregistration section 6: fixed before the first external "
               "arm runs. A later change to any entry turns held-out into "
               "in-sample.",
        "files": {f: sha256_file(HERE / f) for f in FROZEN_FILES},
        "thresholds": thresholds,
        "catalogue": {
            "types": [t["id"] for t in phi.CATALOG],
            "order_is_priority": True,
            "anchors": ["MLRC v2.0 (arXiv:2003.12206)",
                        "NeurIPS Paper Checklist",
                        "PRISMA 2020 (BMJ 2021;372:n71)"],
            "layers": list(phi.LAYERS),
            "type_to_layer": dict(phi.TYPE_TO_LAYER),
            "stages_reached": sorted(set(phi.TYPE_TO_LAYER.values())),
        },
        "development_substrate": {
            "run": str(sub) if sub else None,
            "N": uni["N"] if uni else None,
            "universe_digest": phi.universe_digest(uni) if uni else None,
            "parse_failure_rate": uni["parse_failure_rate"] if uni else None,
            "not_recorded_rate": uni["not_recorded_rate"] if uni else None,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    out = HERE / "FREEZE.json"
    cur = build()

    if a.emit:
        out.write_text(json.dumps(cur, indent=2) + "\n")
        print(f"wrote {out.name}")
        print(f"  types    : {', '.join(cur['catalogue']['types'])}")
        print(f"  stages   : {len(cur['catalogue']['stages_reached'])} of "
              f"{len(cur['catalogue']['layers'])}")
        print(f"  N        : {cur['development_substrate']['N']}")
        print(f"  universe : {cur['development_substrate']['universe_digest'][:16]}")
        return 0

    if a.verify:
        if not out.exists():
            print("FREEZE.json missing -- nothing to verify against")
            return 1
        old = json.loads(out.read_text())
        drift = []
        for section in ("files", "thresholds", "catalogue", "development_substrate"):
            if old.get(section) != cur.get(section):
                for k in set(old.get(section, {})) | set(cur.get(section, {})):
                    if old.get(section, {}).get(k) != cur.get(section, {}).get(k):
                        drift.append(f"{section}.{k}")
        if drift:
            print("INSTRUMENT MOVED since freeze:")
            for d in drift:
                print(f"  {d}")
            print("\nIf any external arm has been run, this is not a fixable "
                  "edit: held-out has become in-sample for that arm.")
            return 1
        print("freeze verified: instrument unchanged")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
