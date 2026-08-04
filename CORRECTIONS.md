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

Replication after submission settled the substance. Quality throughout
these comparisons is the paper's own measure: the mean of the best
verified score per want, taken over the nine wants that any run ever
served. With the replication wave complete, thirty-one independent
coordinator-team runs (the twenty-four-run wave plus the seven earlier
independent team runs) measure the team arm at mean 0.6299 on that
measure, against the market arm's 0.5934 over its nine societies. A
two-sided rank test finds no significant difference (Mann-Whitney
p = 0.38; neither one-sided direction reaches significance, p = 0.82
and p = 0.19). On delivered quality the two arms are a statistical
tie. Every sentence comparing market societies to coordinator teams is
therefore removed.
The single-mind comparison stands: all four market societies scored
above the single long-context mind (0.460 to 0.758 against 0.377), the
best at twice its score and twice its coverage.

## 2. The profession census is re-derived (Section 4, abstract, Figure 1)

The census instrument counts each learner's most frequent settled
delivery as its trade. When two tools tied for most frequent, the
instrument broke the tie in whatever order the interpreter's hash seed
happened to impose, so repeated runs returned different counts. A
one-line change, sorting the tied set before the pick, makes the
instrument deterministic: every run now returns the same counts (see
the reproducibility note in the README and
`instruments/converged_division.py`). Re-derived with the fixed
instrument, over the seventeen societies old enough to census, meaning
their last hundred settled deliveries all fall inside a converged
window: eleven to twenty distinct trades per society, evenness 0.91 to
0.98. The submitted reading of a
perfect division, twenty-three trades over twenty-three learners at
evenness 1.00, does not reproduce; the widest verified division is
twenty trades over twenty-three learners at evenness 0.98. Figure 1 is
rebuilt from the deterministic output.

## 3. The complete-panel transmission claim is removed from abstract and body

The submitted abstract attached "(exact one-sided p = 1/70)" to the
reconstruction claim, and the submitted body asserted the result
outright ("all eight seeded daughters sit nearer their own source
than the other"). That probability belongs to the panel's complete
test, the one that asks whether every one of the eight seeded
daughters ends nearer its own source society than the other source, a
split that has probability 1/70 by chance. The registered read of
that complete test has not been taken. The corrected version removes
the result claim from the abstract and the body both. What the body
keeps is supported: the registered design (stated with its 1/70
success criterion, as registration), the first reads of the panel,
and a confirmation wave, a second pre-registered batch of daughters
and controls, at exact rank-sum p = 1/126. The complete-panel result
will be claimed only when its registered read is taken.

(Amended 4 August: the first version of this note said the submitted
body did not report the complete test. It did, and the first
corrected build retained it; both are now removed, and the corrected
PDF is rebuilt. Same-day second amendment: the abstract's opening of
the reconstruction claim now credits "reconstruction experiments"
rather than "a reconstruction panel", since the supported evidence is
the first reads and the confirmation wave, not the complete panel;
and a stale census count in the limitations section, eleven, is
corrected to the seventeen of the re-derivation.)

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
