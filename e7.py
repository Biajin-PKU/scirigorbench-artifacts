#!/usr/bin/env python3
"""E7 -- can an arm game the leaderboard by padding its universe?

The attack needs no cleverness, which is why it has to be checked before
publishing a leaderboard rather than after. An agent that emits many trivially
checkable claim/artifact pairs -- "we used seed 1", "we used seed 2" -- inflates
N. Column 1 rewards that directly, as better recording discipline. The question
is what it does to column 3.

It is arithmetic rather than simulation, and the answer is exact:

    p = P[Hyp(N, K, A) >= X],   e = kappa * p^(kappa-1)

Holding K, A and X fixed and growing N makes the same number of hits less and
less likely by chance, so p falls and e rises. The adjudicator has not improved.
Only the denominator moved.

    python3 e7.py
"""

import json
import math
import pathlib
import sys
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e1 import calibrate, hyper_sf  # noqa: E402  the certificate, unchanged

BASE_N = 159          # measured universe of the development arm
K = A = X = 3         # its measured certificate line at K=3: A=K and X=K


def main():
    print("E7 -- universe padding against the certificate\n")
    print(f"held fixed: K={K}  A={A}  X={X}   (the development arm's own line)")
    print(f"{'N':>8} {'padding':>9} {'p':>12} {'e':>12} {'x vs base':>10}")
    print("-" * 56)

    rows, base_e = [], None
    for N in (BASE_N, 300, 600, 1200, 2412, 5000):
        p = float(hyper_sf(X, N, K, min(A, N)))
        e = calibrate(p)
        base_e = base_e or e
        pad = N - BASE_N
        rows.append({"N": N, "padding": pad, "p": p, "e": e,
                     "inflation": e / base_e})
        print(f"{N:>8} {pad:>9} {p:>12.3g} {e:>12.6g} {e / base_e:>9.1f}x")

    infl = rows[-2]["inflation"]
    print(f"""
Padding the universe from {BASE_N} to {rows[-2]['N']} multiplies the e-value by
{infl:.0f} without the adjudicator catching one additional planted claim. The
attack is available to any arm that can emit checkable trivia, and emitting
checkable trivia is cheap.

Two properties of the construction make it worse, not better:

  * Column 1 rewards the padding directly. An arm that floods N reads as having
    the best recording discipline on the board.
  * The padded claims are genuinely checkable, so a uniform draw lands in the
    padding and the arm's own adjudicator resolves them easily. Nothing about
    the attack requires the arm to lie.

What blunts it, and what does not:

  * Stratifying the draw helps only in proportion to how many stages the padding
    reaches. Padding one stage leaves the others' K_l untouched, so an arm must
    pad every stage to move every stratum -- more work, not a different kind.
  * Reporting e alone does not blunt it at all. e is not comparable across
    arms with different N, and this is the demonstration.

⇒ The leaderboard must never present an e-value without the N it was computed
  against, and cross-arm comparison of column 3 requires a quantity that is
  invariant to universe size. That is a defect in the specification of the
  leaderboard, found before the leaderboard existed, which is the only good
  time to find it.""")

    (HERE / "e7_padding.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
