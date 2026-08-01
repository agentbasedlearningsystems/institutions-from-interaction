# Corrections to the submitted paper, 1 August 2026

The version of record as submitted is `paper/CSS2026_paper_78_as_submitted.pdf`.
The corrected version is `paper/CSS2026_paper_78_corrected.pdf`. The
corrections came from an internal audit that recomputed every reported
figure from the raw run records. They are listed here so a reader can
check each one.

## 1. The arms comparison is withdrawn (Section 10, abstract, conclusion)

The submitted paper compared four market societies against three
non-market runs and reported that all four sat above every non-market
collective, exact one-sided p = 1/35. Two of those three runs were the
same run: the second coordinator-team run was a byte-identical replay of
the first, produced by a response cache at temperature zero. The
independent count was two, so the design supported at most p = 1/15,
and the printed 1/35 was wrong at submission.

Replication after submission settled the substance. Thirty independent
coordinator-team runs measure the team arm at mean 0.6289 on the same
nine-want measure, above the market arm's 0.5934. Every sentence
comparing market societies to coordinator teams is therefore removed.
The single-mind comparison stands: all four market societies scored
above the single long-context mind (0.460 to 0.758 against 0.377), the
best at twice its score and twice its coverage.

## 2. The profession census is re-derived (Section 4, abstract, Figure 1)

The census instrument broke frequency ties in interpreter hash order,
so repeated runs returned different counts (see the reproducibility
note in the README and the one-line fix in
`instruments/converged_division.py`). Under the deterministic
instrument: seventeen censusable societies, eleven to twenty distinct
trades per society, evenness 0.91 to 0.98. The submitted reading of a
perfect division, twenty-three trades over twenty-three learners at
evenness 1.00, does not reproduce; the widest verified division is
twenty trades over twenty-three learners at evenness 0.98. Figure 1 is
rebuilt from the deterministic output.

## 3. The abstract's transmission probability is removed

The submitted abstract attached "(exact one-sided p = 1/70)" to the
reconstruction claim. That probability belongs to the panel's complete
unanimous-split test, which the body does not report; the body reports
first reads and a registered confirmation wave (exact rank-sum
p = 1/126). The parenthetical is removed until the panel completes.

## 4. Sign informativeness range

The submitted range of 0.68 to 1.55 bits does not reproduce at its
upper end; no current society reads above 1.05. Corrected to 0.69 to
1.05 bits over twenty-nine societies. The median, 0.74, is unchanged.

## 5. Settings-variant count

Ten of twenty-two testable societies becomes twelve of twenty-seven at
the current read. The stated conclusion, a recurring local convention
rather than a population-wide law, is unchanged.

## 6. Employment sentence aligned with its own figure

The text said the learner-to-learner share rises late in seven of
twenty-seven ledgers; the figure caption says ten. The recount supports
the caption; the text now says ten.

Results that reproduce exactly and are unchanged: the hidden-capability
experiment in all three arms, the wholesale-supplier result and its
out-of-sample replication, the solitary-control comparison, the
settlement-order result, and the employment range.
