# SciRigorBench — code and artifacts

Anonymised release accompanying the submission.

## What is here

    phi.py                 the instrument: claim universe, artifact index, binding, planting
    freeze.py / FREEZE.json  frozen manifest; `python freeze.py --verify` fails if any
                           digest, threshold, catalogue entry or the universe digest has
                           moved since the first external arm was run
    m1.py                  manuscript-only adjudicator
    e1.py .. e22.py        one file per experiment (some numbers unused: e3, e6, e19)
    e*_*.json              the measured outputs those scripts wrote
    verify_numbers.py      recomputes headline values and fails if the manuscript disagrees
    arms/                  five external arms' extraction drivers and their phi outputs
    contracts/             frozen artifact-scope declarations
                           development substrates ship as substrate-NN.json; the original
                           names are committed by sha256 in FREEZE.json release_mapping
    D3-recording-contract.md  the recording contract in full
    paper/                 the manuscript (main.tex / main.pdf) and figures

## Running

    python freeze.py --verify
    python e1.py                  # two adjudicators on one universe
    python e12.py                 # floor sweep over the five external arms
    python e21.py                 # matched pair on the fixed package of e15
    python e22.py                 # coverage attribution / tie structure
    python verify_numbers.py      # every headline number vs main.tex

External corpora are not redistributed. They are
`chchenhui/mlrbench` (MIT; AI Scientist-v2 and MLR-Agent arms),
`ARA-Labs/Agent-Native-Research-Artifact` (MIT),
the AI Scientist-v1 release, and the ICML-2026 reproduction challenge
submissions on the Hugging Face Hub, each used at the commit or revision
recorded in the artifact manifest.

## Redaction

`FREEZE.json` redacts absolute paths on the development substrates and carries a
`sha256` commitment to each original string so the redaction is checkable after
de-anonymisation. Contract files whose original names identified an internal
generator are renamed `substrate-NN.json`; `release_mapping` records the
original name and its sha256. `reportable_outputs` in every contract is
byte-identical to the frozen original.
