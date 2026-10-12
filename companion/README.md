# Companion material

Everything behind the counts in *Autonomous Perception Through Emergent Signs: The Conditions for an Emergent
Division of Labour Among Independent Learning Agents*, a paper under double-blind review. The documents in
`../paper/` are the anonymous versions, and this material carries no author information either.

- `simulation/` — the engine: the market, the agents, the one draw by sign, settlement, payment and learning.
- `scripts/` — the world generators, and the readers that computed every measurement from a record.
- `reads.zip` — every reading file behind every number, one text file per world and reading. A name ending
  `_bN` is the reading from the record cut at round N; `_final` is the reading of the whole record at the end of
  the run; `mlcomp_` files are the composition readings that follow each delivered program back to the agents
  that made its parts. Superseded readings are kept apart under `_lagged/` and no number comes from them.

The supplement in `../paper/supplement_anonymous.pdf` defines every measurement by the computation that produces
it, so a table can be remade from a record.

Machine paths in the reading files have been replaced by `/home/user/...`.
