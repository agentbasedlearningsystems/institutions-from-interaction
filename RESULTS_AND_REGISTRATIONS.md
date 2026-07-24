# Results tables and registration ledger

Supplement to "Institutions from Interaction: The Emergence and
Transmission of a Division of Labor in Coevolutionary Societies"
(CSSSA 2026) and its ODD model description. This file is part of the
public reproduction kit; it is not part of the conference submission.
Every table is computed by the instruments in `instruments/` from the
run outputs of the societies in `configs/`.

# Complete results data

This section carries the full per-society data behind every number in
the companion paper, so that any claim can be traced to its society.
All societies named here are the standard cohort (the single basic
scenario), its always-on variants of the same scenario, and the
solitary controls; every statistic is computed by the instruments
defined in the ODD's measurement section and shipped in
`instruments/`.

## Table A1: Per-society evolutionary results (all legitimate societies, July 21 late-evening read; live societies read to their log end)

| society | trades | learners | even. | LL%val | LLrise | settle | MI | fineDelta | finep | junkE/U |
|---|---|---|---|---|---|---|---|---|---|---|
| alwayson_seed210 | 15 | 24 | 0.94 | 1.5 | no | holds | 0.75 | +0.017 | 0.003 | 43/-- |
| alwayson_seed211 | 13 | 16 | 0.98 | 0.0 | no | holds | 0.75 | +0.038 | 0.003 | 0/0 |
| alwayson_seed212 | 17 | 22 | 0.97 | 19.8 | yes | holds | 0.78 | +0.000 | 0.183 | 5/-- |
| alwayson_seed213 | 4 | 4 | 1.00 | 0.0 | no | holds | 0.68 | -0.007 | 1.000 | 0/-- |
| alwayson_seed214 | 0 | 0 | 0.00 | 0.0 | no | young | 1.24 | - | - | - |
| alwayson_seed215 | 2 | 2 | 1.00 | 0.0 | no | holds | 0.84 | - | - | 0/-- |
| basic_seed216 | 0 | 0 | 0.00 | 0.0 | no | holds | 0.79 | - | - | 0/-- |
| basic_seed217 | 0 | 0 | 0.00 | 0.0 | no | young | 1.05 | - | - | - |
| basic_seed218 | 1 | 2 | 0.00 | 0.0 | no | holds | 0.80 | -0.015 | 1.000 | 0/-- |
| basic_seed219 | 13 | 23 | 0.97 | 6.6 | yes | holds | 0.72 | +0.017 | 0.003 | 25/-- |
| basic_seed220 | 17 | 24 | 0.95 | 8.5 | yes | holds | 0.69 | +0.022 | 0.003 | 48/-- |
| basic_seed221 | 3 | 3 | 1.00 | 0.0 | no | holds | 0.72 | +0.028 | 0.003 | 0/-- |
| basic_seed222 | 16 | 20 | 0.98 | 1.4 | no | holds | 0.70 | -0.042 | 1.000 | 32/-- |
| basic_seed223 | 15 | 22 | 0.98 | 1.9 | yes | holds | 0.74 | -0.025 | 1.000 | 46/0 |
| basic_seed224 | 19 | 24 | 0.97 | 3.6 | yes | holds | 0.72 | -0.003 | 0.993 | 31/-- |
| basic_seed225 | 15 | 19 | 0.98 | 0.0 | no | holds | 0.74 | +0.027 | 0.003 | 0/-- |
| basic_seed226 | 16 | 17 | 0.99 | 3.1 | yes | holds | 0.72 | -0.024 | 1.000 | 8/-- |
| basic_seed227 | 20 | 23 | 0.99 | 1.3 | no | holds | 0.70 | +0.009 | 0.003 | 41/0 |
| basic_seed228 | 16 | 24 | 0.95 | 2.1 | yes | holds | 0.75 | -0.019 | 1.000 | 56/0 |
| basic_seed229 | 18 | 23 | 0.96 | 1.4 | no | holds | 0.72 | +0.012 | 0.003 | 50/-- |
| basic_seed230 | 16 | 24 | 0.91 | 3.8 | yes | holds | 0.73 | -0.029 | 1.000 | 57/-- |
| basic_seed231 | 17 | 23 | 0.96 | 5.0 | no | holds | 0.73 | +0.030 | 0.003 | 56/0 |
| basic_seed232 | 8 | 10 | 0.97 | 4.7 | yes | holds | 0.78 | -0.010 | 1.000 | 6/-- |
| basic_seed233 | 11 | 14 | 0.94 | 0.0 | no | holds | 0.73 | +0.018 | 0.003 | 6/-- |
| basic_seed234 | 10 | 12 | 0.98 | 0.0 | no | holds | 0.76 | -0.024 | 1.000 | 5/-- |
| basic_seed235 | 9 | 10 | 0.98 | 10.6 | yes | holds | 0.73 | -0.010 | 1.000 | 5/-- |
| basic_seed236 | 0 | 0 | 0.00 | 0.0 | no | holds | 0.82 | - | - | 0/-- |
| basic_seed237 | 0 | 0 | 0.00 | - | - | young | 1.55 | - | - | - |
| solo_seed200 | 0 | 0 | 0.00 | 0.0 | no | holds | - | - | - | - |
| solo_seed201 | 0 | 0 | 0.00 | 0.0 | no | holds | - | - | - | - |
| solo_seed202 | 0 | 0 | 0.00 | 0.0 | no | holds | - | - | - | - |

 Columns: distinct trades, true learners, and evenness
(merged across archived and current report segments, replayed rows
deduplicated); learner-to-learner share of end-window payment value
(percent) and whether it exceeds the lifetime share; settle-order
verdict; displayed-sign informativeness MI (bits); finest-grain
within-versus-between family sign-similarity delta and permutation
p; junk share entry/upper tier (percent). Sign columns are the
displayed-sign instruments on every society with enough listings.

## Table A2: Task-family sign geometry (displayed signs, July 21 late-evening read)

| society | rep/anom sellers | separation | perm p |
|---|---|---|---|
| alwayson_seed210 | 9/1 | sparse | |
| alwayson_seed211 | 6/9 | -0.0156 | 0.639 |
| alwayson_seed212 | 5/3 | -0.0040 | 0.595 |
| alwayson_seed213 | 10/8 | +0.0312 | 0.116 |
| alwayson_seed214 | 9/10 | -0.0297 | 0.820 |
| alwayson_seed215 | 11/8 | +0.0775 | 0.006 |
| basic_seed216 | 7/7 | +0.0206 | 0.327 |
| basic_seed217 | 15/6 | +0.0669 | 0.024 |
| basic_seed218 | 11/11 | +0.0055 | 0.416 |
| basic_seed219 | 6/2 | -0.0446 | 0.808 |
| basic_seed220 | 7/3 | +0.0044 | 0.429 |
| basic_seed221 | 7/3 | -0.0155 | 0.533 |
| basic_seed222 | 3/1 | sparse | |
| basic_seed223 | 7/3 | +0.0392 | 0.166 |
| basic_seed224 | 6/3 | +0.0662 | 0.117 |
| basic_seed225 | 6/6 | +0.0112 | 0.337 |
| basic_seed226 | 8/2 | +0.0124 | 0.387 |
| basic_seed227 | 7/5 | +0.0359 | 0.209 |
| basic_seed228 | 8/3 | -0.0587 | 0.886 |
| basic_seed229 | 7/6 | -0.0383 | 0.805 |
| basic_seed230 | 7/2 | +0.0313 | 0.300 |
| basic_seed231 | 9/7 | -0.0361 | 0.852 |
| basic_seed232 | 6/4 | +0.0459 | 0.164 |
| basic_seed233 | 6/4 | +0.0043 | 0.381 |
| basic_seed234 | 9/5 | -0.0075 | 0.545 |
| basic_seed235 | 11/5 | -0.0214 | 0.721 |
| basic_seed236 | 24/7 | +0.0109 | 0.312 |
| basic_seed237 | 13/11 | +0.0110 | 0.327 |

 Direction summary: separation positive in 16 of 26; sign
test one-sided p = 0.16; median separation +0.0082. Not yet
emerged at these ages, leaning positive.

## Table A3: Mixed-capability arm A, per society

| society | form | Haiku mean rank | Sonnet mean rank | Opus mean rank |
|---|---|---|---|---|
| m1 | market | 9.0 | 4.5 | 1.5 |
| m2 | market | 8.0 | 5.5 | 2.0 |
| m3 | market | 9.0 | 4.5 | 1.5 |
| t1 | team | 5.4 | 6.0 | 7.5 |
| t2 | team | 6.6 | 5.8 | 5.0 |
| t3 | team | 6.2 | 5.8 | 6.0 |

 Ranks within society (1 = top) on net income (market) or
score-mass (team). Hidden cast per society: 5 Haiku, 4 Sonnet, 2
Opus. Pooled capable-versus-Haiku rank gap: market -4.89
(p < 10^-4), team -0.12 (p = 0.47).

## Table A4: Language-model societies (full mechanism)

| society | wholesaler | comp.\ sales | consumers | share | net income |
|---|---|---|---|---|---|
| school3_a | A3 | 194 | 10 | 51% | +2045 |
| school3_b | A4 | 98 | 11 | 55% | +2262 |
| school3_c | A3 | 59 | 9 | 39% | +732 |
| school3_c2 | A11 | 152 | 10 | 42% | +1933 |

 In every society the component wholesaler is also the top
net earner. Cross-corpus generalization rung: served in all four,
best scores 0.747-0.754. Deduplication rung: zero sales in all four
(rational deferral; agents' planning text cites capital and
unfamiliar scoring). Single-mind control: one agent, 300 periods,
same demand sheet (run `cw_dial3_1x300`); coordinator-team
controls: `cw_orch_clean_a/b`.

## Table A5: Reconstruction panel registry and reads (through the July 22 midday declared read)

| world | seeded from | status | read |
|---|---|---|---|
| cr_3x270 | X = basic-227 (concentrated) | growing | pending age 300 |
| cr_3x271 | X | interim read at age 381 | fidelity +0.696 |
| cr_3x272 | X | interim read at age 338 | fidelity +0.825 |
| cr_3x273 | X | growing | pending age 300 |
| cr_3y274 | Y = alwayson-211 (dispersed) | read at age 497 | fidelity +0.619 |
| cr_3y275 | Y | read at age 386 | +0.684 |
| cr_3y276 | Y | read at age 364 | +0.636 |
| cr_3y277 | Y | read at age 422 | +0.575 |
| cr_3n278 | none (control) | read at age 437 | simY 0.100, simX 0.000 |
| cr_3n279 | none (control) | read at age 396 | simY 0.133, simX 0.000 |

 Sources chosen by maximal practice-profile divergence among
the eight eldest societies (total variation 0.913, two shared trades;
frozen tools-only standing-state censuses: X 19 deliveries / 7 tools
/ 20 uses, Y 8 / 7 / 10; the sources' own mutual similarity 0.05).
X-daughters iterate roughly seven times slower because X's standing
state includes computationally costly tools (TSNE, KNNImputer), a
property of the state itself, declared before any read. The X-side
interim reads were each authorized singly and disclosed as interim;
the panel-complete unanimous-split test (exact one-sided p = 1/70)
remains the X-side primary and reports in the final version. The
x272 census at its read was exactly frozen X's seven tools; x271 held
six of the seven plus three additional low-count tools.

## Table A5b: Confirmation wave, finalized at the July 22 midday declared read

Five fresh Y-daughters against five fresh unseeded controls, seeded
from the same frozen Y census as the pilot, single declared read at
age 300 or more, qualifiers stopped on qualification with data frozen
in place: y280 = 0.655 (age 356), y281 = 0.600 (406), y282 = 0.657
(478), y283 = 0.661 (362), y284 = 0.582 (604); controls n285 = 0.000
(524), n286 = 0.174 (378), n287 = 0.100 (347), n289 = 0.000 (328).
The wave finalized at nine qualifiers: full separation, five against
four, exact rank-sum p = 1/126. The tenth world (n288) had reached
age 224 at the read and keeps aging; any later qualification is
final-version material only, and the shortfall from ten is
pace-driven and disclosed. The pilot daughters and controls of Table
A5 are reported alongside and never pooled into the confirmation
statistic.

## Pricing snapshot (July 22 midday declared read; registered predictions stand unconfirmed)

Across the twenty-eight standard societies, boards first to latest:
composite-tier midpoint premium positive in 17 of 28 (sign test
p = 0.172), below the registered flip threshold, so the
composite-premium prediction remains registered and unconfirmed;
premium widening since birth in 19 of 28 (p = 0.044); entry-tier
midpoints falling in 20 of 28 (examples: 55 to 40, 47 to 36, 50 to
38); entry median 47.00, composite median 49.10. A within-buyer check
of quality-hunger alpha against offer midpoint was null at the
July 21 read (positive in 12 of 28, median Spearman rho = -0.006):
alpha does not yet steer midpoints.

## Table A6: The organizing-mechanism comparisons (July 21 morning)

Same demand sheet throughout. Single long-context mind (300 periods):
mean best-per-want 0.377, 4/9 wants served. Coordinator teams: 0.376
and 0.376, 5/9. Market societies: 0.460-0.758, up to 8/9; all four
above all three non-market collectives, exact one-sided p = 1/35.
Paired mixed-capability quality (matched casts): both forms reach the
task ceilings (0.75-0.76; want-level sign test p = 0.79); only the
market also sorts reward (arm A). Evolutionary solitary controls: at
equal rounds (400), both below all eleven comparable societies
(0.02/0.06 against 0.17-0.29; exact p = 1/78); at equal
learner-effort, comparable (0.17/0.29 in the 0.20-0.34 band).
Wholesaler criterion, first unseen pass: 2/3 arm A markets (the third
at 26 percent share against the 30 percent bar); cross-corpus rung
3/3 unseen at 0.758-0.763; deduplication 0 sales in all 7 societies
examined; in all three unseen markets the wholesaler was the hidden
top-capability agent that also topped income (descriptive).
Employment against quality: raw Spearman rho = 0.330 (p = 0.021,
n = 26) but age-confounded (partial approximately  -0.16), reportable
only with the age caveat.

## Table A7: Staged-sign arms, final (July 21 evening)

All twelve societies landed at the 20-period checkpoint (pair 1
truncated at periods 10-13 by a machine failure independent of
outcomes; counts included). Arm B uptake: market 18/33 workers bought
costly sign dimensions, hierarchy 0/33 (Fisher exact
p = 1.5 x 10^-7); bought-spend against net income
r = -0.26 (n = 33). Arm C within-tier stuck-group contrast,
stratified pairs pooled: market -0.65 ranks (p = 0.27), hierarchy
-0.10 (p = 0.88), a bounded null. Wholesaler criterion, unseen
set at nine societies: 6/9 pass, two near misses; cross-corpus rung
9/9; top supplier equals top earner in 7/9; the role is held by large
and small hidden models about equally. Deduplication: zero sales in
all thirteen societies examined. Across all nine market societies:
mean within-society correlation between hidden tier and income rank
-0.814 (permutation p <= 10^-5, 100{,}000 reps); top earner a
hidden large-capability agent in 8/9 (binomial p = 9 x
10^{-6}), the ninth the small-tier wholesaler, whose entire net
income came from intermediate sales to six agent buyers.

# Registration and amendment ledger

Chronological, as declared before each read;
nothing here was written after its read.

1. **Utility pricing rule** (July 20): genome-learned offer
bands on both sides; band double auction (no intersection, no sale;
price at the intersection midpoint); quality scaling by score over
frontier retained; buyer fitness U = Delta^{alpha} m^{1-alpha};
budgets authored (degree of want), prices never authored. Registered
predictions P1--P4 (part prices fall and settle; composite shares
stay high; premium positive across the alpha spread; tempo order),
confirmatory read at cohort maturity.

1. **Mixed-capability arms** (July 20): hidden cast of eleven
(5 Haiku / 4 Sonnet / 2 Opus), identical across paired forms; no
individual recognition anywhere; arm A fully free signs; arm B adds
bought sign dimensions (cost per unit intensity per round, zeroed if
unaffordable); arm C adds one stuck dimension, birth-assigned +/-
1, identical capability distributions across groups. Team-form
scrip wage per scored delivery funds bought signs.

1. **Arm A stopping rule** (July 20, evening): read at
period 20; one interim peek at one-third distance disclosed.

1. **Reconstruction panel 3** (July 20, night): sources by
maximal divergence; census equals standing state (the last five
iterations of the source's live segment); seeding equals conditional
probabilities at every level, ten percent of the tool space uniform
plus ninety percent census-proportional; tools-only counting (trades
are what is traded); data and instrument branches at natural uniform
weights; defaults credit the function, never its settings children;
references frozen; read at daughter age 300; unanimous-split primary
(exact p = 1/70); nulls define drift; diagnostic gap reported
alongside. Two same-night amendments (window and counting) both
predate every read.

1. **Prior reconstruction panel** (July 18-19): retired
before any reported read; its family-root balancing erased the
family-level census axis (verified by a newborn-decode test) and a
seed-thickness asymmetry favored one source; its reads appear
nowhere in the paper.

1. **Confirmation wave** (July 21, morning, declared before
launch): five fresh Y-daughters from the same frozen census and five
fresh unseeded controls; registered one-sided exact rank-sum test on
standing-state similarity to frozen Y at age 300 or more; the pilot
daughters and controls are reported alongside and never pooled;
qualifier shortfalls at the declared read are pace-driven and
disclosed.

1. **Arms B and C expansion** (July 21, morning, declared
before launch): two further pairs per arm, both forms, twenty
periods, same hidden casts; new arm C pairs stratify stuck-sign
births so capability tiers split evenly between markers; the first C
pair's accidental capability-confounded split is disclosed and
excluded from the pooled contrast; declared pooled tests: uptake by
form (Fisher exact), spend against income (market form), stuck-group
within-tier contrasts in each form. The first pair's crash
truncation at periods 10-13 is disclosed and its data enters uptake
analyses with the truncation stated.

1. **Wholesaler, cross-corpus, and deduplication unseen
tests** (July 21, declared before the unseen pass): criterion
calibrated on the four school societies and applied unseen to the
mixed-capability market societies; exact binomial against a null of
at most half; the twenty-period, eleven-agent moderator declared
before looking.

1. **Nine-society income gradient** (July 21, evening): arm
A's declared tier-income statistic computed over the complete
market-society population after arm A's test, disclosed as a
post-test extension wherever it appears.

1. **X-side interim reads** (July 22): two single-daughter
fidelity looks (x272, then x271) each authorized before the look and
disclosed as interim; the panel-complete unanimous-split primary
remains untouched.

# Code and data availability

The engine, every society configuration, the language-model market
harness, every analysis instrument, the hidden capability casts, and
the frozen reconstruction references are public at
https://github.com/agentbasedlearningsystems/institutions-from-interaction
(Agent Based Learning Systems,
https://agentbasedlearningsystems.com). Full run outputs
(reports, boards, payment ledgers, and transcripts, tens of
gigabytes) are retained in full and available on request.
Language-model transcripts are cached by prompt, so completed
societies replay deterministically at zero cost. Hidden capability
casts live in a truth directory never shown to agents. Every
instrument is also defined by its computation in the measurement
section of this document.

# Exhibits from the ledgers

**Employment exhibits, receipt by receipt.** The largest single
employment relation recorded so far: one learner paid another 531
tokens across eighteen transfers for components feeding its own
sales. The deepest recorded structure is a five-level cascade set off
by a single sale: a buyer paid a learner for a finished product; that
learner paid a second for a component; the second paid a third for a
deeper component; and the third paid two suppliers of its own. One
purchase, four learners employed beneath the seller, subcontractors
with subcontractors. A second recorded shape is the middleman: one
learner selling to seven distinct buyers in a single instant while
paying its own wholesale supplier.

**Improvement decomposition.** Junk (score at or below 0.05)
concentrates at the entry tier: median 6.5 percent of entry-tier
settlements across the twenty-five societies measured, ranging to
57 percent in the heaviest, while the tiers above read median zero.
The frontiers sit at the practical ceilings of their tasks: anomaly
detection at 0.96 to 0.98, classification at 0.78 to 0.88
(comparable to a competently hand-built pipeline on these corpora),
embeddings near what these tools express. The two rungs beyond those
ceilings, ground-truth-matched clustering and multi-family ensembles,
remain unclimbed after thousands of iterations: a long climb still in
progress, not a freeze. The language-model societies, whose buyers'
expectations ratchet upward by construction, improved exactly where
headroom remained.

**Speed.** The reported structures are visible within the first
three hundred iterations (wall-clock hours) at settlement rates of
1.2 to 1.4 per iteration.

## Table A8: Converged-state census (July 22 afternoon; her convergence ruling)

Window: each society's last 100 settled deliveries (the converged
standing state). Societies with a full or near-full window (>= 50
recent deliveries), every one dividing:

| society | trades | learners | evenness | window |
|---|---|---|---|---|
| basic_seed220 | 19 | 22 | 0.99 | 100 |
| basic_seed227 | 18 | 23 | 0.98 | 100 |
| basic_seed230 | 17 | 20 | 0.98 | 100 |
| alwayson_seed212 | 17 | 22 | 0.98 | 63 |
| basic_seed222 | 17 | 22 | 0.98 | 92 |
| alwayson_seed210 | 15 | 21 | 0.98 | 100 |
| basic_seed219 | 15 | 23 | 0.92 | 100 |
| basic_seed229 | 15 | 21 | 0.97 | 100 |
| basic_seed223 | 15 | 21 | 0.97 | 100 |
| basic_seed228 | 14 | 20 | 0.97 | 100 |
| basic_seed231 | 14 | 18 | 0.98 | 100 |
| basic_seed224 | 13 | 22 | 0.95 | 100 |

Converged junk (same window, 25 societies measured): entry-tier
median 6.5% of settlements, upper tiers median 0%. The remaining
societies' converged windows are still filling; instrument:
`instruments/converged_division.py`, `instruments/converged_junk.py`.

## Measurement standard (July 24, standing for all future reads)

**The standing state** of a society is its last one hundred settled
deliveries, and every descriptive instrument reads inside that span:
the profession census (distinct professions and their practitioners),
junk decomposition, the contested-roles count, employment (payments
within the span), and sign informativeness (boards covering the span).
Delivery-mass windows make societies of different speeds comparable.
Two fixed exceptions: reconstruction fidelity reads keep their
last-five-iterations census (set before any daughter data existed),
and lifetime statistics are always labeled "lifetime."

**Terminology**: a *profession* is a distinct kind of work practiced
(what the census counts); a *delivery* is one settled sale; a
*payment* is one ledger transfer. "Trades" in earlier tables means
professions.
