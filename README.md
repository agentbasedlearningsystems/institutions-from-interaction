> **Correction notice, updated 4 August 2026.** The originally
> submitted paper contained errors that an internal recomputation of
> every reported figure found after submission; the paper was
> RESUBMITTED in corrected form, and the resubmitted version of record
> incorporates the corrections listed in
> [CORRECTIONS.md](CORRECTIONS.md). The original submission is kept as
> [paper/CSS2026_paper_78_as_submitted.pdf](paper/CSS2026_paper_78_as_submitted.pdf)
> for comparison. The newest corrected build,
> [paper/CSS2026_paper_78_corrected.pdf](paper/CSS2026_paper_78_corrected.pdf),
> additionally carries four same-day polish edits made after the
> resubmission upload (listed at the end of the corrections file). The
> largest correction: the market-versus-coordinator-team comparison
> counted a cache replay as an independent run and is withdrawn; the
> completed replication reads as a statistical tie.

# Institutions from Interaction — code and reproduction kit

Code, configurations, and analysis instruments for the paper
"Institutions from Interaction: The Emergence and Transmission of a
Division of Labor in Coevolutionary Societies" (CSSSA 2026) and its
ODD companion document.

Agent Based Learning Systems — https://agentbasedlearningsystems.com

## Layout

- `simulation/` — the evolutionary-substrate engine (SnetSim): agents
  with CMA-ES learners trading under the designed market mechanism.
- `configs/` — every society in the paper. `bigtree_basic_*` are the
  standard-scenario societies (seeds 200-237), `bigtree_alwayson_*`
  a registered variant, `bigtree_solo_*` the solitary controls;
  `cr_3x*`/`cr_3y*`/`cr_3n*` are the coevolutionary-reconstruction
  daughters and unseeded controls.
- `llm_market/` — the language-model-substrate harness: the same
  market mechanism with a language model inside each agent
  (`hetero_sdss.py` runs the mixed-capability arms; `llm_backend.py`
  is the provider seam with a transcript cache and a spend ledger).
- `instruments/` — every analysis in the paper: division census
  (`merged_division.py`), employment (`employment_endwindow.py`),
  settle order (`settle_order2.py`), displayed-sign informativeness
  and finest grain (`sign_battery.py`), task-family geometry
  (`geometry_displayed.py`), junk decomposition (`junk_decomp.py`),
  reconstruction seeding (`seed_from_source_v5.py`) and fidelity
  reads (`peek_x272.py`, the wave reader), and the staged-sign
  analyses (`bc_analysis.py`).
- `truth/` — the hidden capability casts for the mixed-capability
  societies (never shown to agents during runs) and the frozen source
  profiles used by every reconstruction fidelity read.
- `figures/` — figure-generation scripts.

## Reproducing the experiments

Evolutionary societies (no API cost):

    python3.12 -m venv venv && venv/bin/pip install -r requirements.txt
    cp configs/bigtree_basic_seed219.json .
    PYTHONPATH=. venv/bin/python simulation/SnetSim.py bigtree_basic_seed219.json

Each society writes `experiments/<name>/` with a delivery report,
metrics, payment ledger, and board logs; the instruments read those
directories. Societies run for days at 4-8 cores each; results in the
paper read every society to its log end.

Language-model societies (API key required; the full mixed-capability
program cost about $376 at list prices):

    export ANTHROPIC_API_KEY=...
    python llm_market/hetero_sdss.py            # arms A/B/C per its header

`llm_backend.py` caches every (model, prompt) transcript to disk, so
re-running a completed society replays deterministically at zero cost;
the original transcript sets are available on request.

Analyses, run against a directory of experiment outputs:

    venv/bin/python instruments/merged_division.py experiments 42 archive_reports
    venv/bin/python instruments/sign_battery.py experiments
    venv/bin/python instruments/employment_endwindow.py experiments

Full run outputs (reports, boards, transcripts; tens of gigabytes) are
not in this repository and are available on request.


## Reproducibility note, 1 August 2026

The profession census (`instruments/converged_division.py`) previously
iterated a Python set, so tied tools resolved in per-process hash order
and the census could return different counts on different runs. It is
now deterministic (`sorted(...)`, one line). Counts computed before this
date are seed-dependent and should be re-derived; under the fixed
instrument the censusable societies read 11 to 20 distinct trades at
evenness 0.91 to 0.98, identical across every seed tested.
