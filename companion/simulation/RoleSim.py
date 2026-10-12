import numpy as np
import random
import cma
import json
import copy
import math
import sys
import os

from simulation.MarketAgent import MarketAgent, stop_cut


class RoleSim(MarketAgent):
    def __init__(self, unique_id, model, message, parameters):
        super().__init__(unique_id, model, message, parameters)

        # RoleSim uses CMA-ES on float vectors, that are then converted
        # to trade plans.  If an initial message is put on the board,
        # this is converted to a floatvec, and is used to initialize cma-es

        vs = self.vector_size()
        self.initialVec = np.random.uniform(low=0.0, high=1.0, size=(vs,))
        # STUCK SIGNS (her SDSS staging, arm C; approved 2026-08-11):
        # the first stuck_sign_dims display dimensions are ASCRIBED -
        # drawn once at birth from the seeded stream, independent of
        # everything (orthogonal to talent BY CONSTRUCTION), never
        # changeable, never entering fitness directly. The world gives
        # the badge meaning or refuses to - that is the experiment.
        # Newborns draw fresh (her seeds-are-initial-state law).
        _k_stuck = int(self.p.get('stuck_sign_dims', 0) or 0)
        if _k_stuck > 0:
            # GROUP badges, not identifiers (her correction 2026-08-11):
            # the agent is dealt into one of a FEW classes from the
            # seeded stream; every class member wears the same fixed
            # badge, so recognition stays impossible and in-group /
            # out-group structure is real. Class deal is independent of
            # everything else - orthogonal to talent by construction.
            import random as _r
            _n_cls = int(self.p.get('stuck_sign_classes', 3) or 3)
            self._stuck_class = int(np.random.uniform(0, _n_cls))
            _rng = _r.Random('stuckbadge:%d:%d' % (
                int(self.p.get('seed', 0)), self._stuck_class))
            self._stuck_sign = [round(_rng.uniform(0.0, 1.0), 3)
                                for _ in range(_k_stuck)]
        # Coevolutionary reconstruction seeding (her program, built
        # 2026-08-10): when the config carries natal_proportions - a
        # parent society's MEASURED behavior proportions - the natal
        # genome's structured fields are sampled to express them, and
        # nothing else transfers: no genomes, no products, no wealth,
        # no expectations (those re-derive from frequency, her ruling).
        # Draws come from the config-seeded stream, so daughters are
        # exactly as reproducible as any launch.
        self.initialVec = self._apply_natal_proportions(self.initialVec)
        if not message:
            # COEVOLUTIONARY RECONSTRUCTION SEED (3 Oct 2026, her Aug 6 rule:
            # seeds are the INITIAL STATE only). seed_messages, keyed by
            # agent id, gives a learner a starting message (a role: its
            # display sign and sell section) decoded exactly as an authored
            # human's mask is; the pinned floats become the CMA-ES starting
            # mean and nothing is pinned afterwards.
            _seeds = self.p.get('seed_messages') or {}
            _seed = _seeds.get(str(unique_id)) if isinstance(_seeds, dict) else None
            if _seed:
                _msk = copy.deepcopy(_seed)
                self.message, float_vec = self.float_vec_to_trade_plan(self.initialVec, _msk)
                self.initialVec = np.array(list(float_vec), dtype=float)
                self.message, float_vec = self.float_vec_to_trade_plan(self.initialVec)
                # A seeded learner may start with a smaller step (seed_sigma), so
                # the seeded role survives the first generations; CMA-ES adapts
                # sigma from there as for everyone else.
                if self.p.get('seed_sigma'):
                    self._seed_sigma = float(self.p.get('seed_sigma'))
            else:
                self.message, float_vec = self.float_vec_to_trade_plan(self.initialVec)
            self.initial_trade_plan = None
        else:

            self.initial_trade_plan = copy.deepcopy(self.message)
            mask = copy.deepcopy(self.initial_trade_plan)
            self.message, float_vec = self.float_vec_to_trade_plan(self.initialVec, mask)
            # SHARED WANT SIGNS AS THE INITIAL STATE ONLY (7 Oct 2026 night, the language-model culture debug;
            # her 6 Aug rule): with humans_shared_want_signs == 'initial', an authored buy row's 'sign' is pinned
            # by this first decode into the starting vector and then dropped from the mask, so the consumer's
            # sought sign evolves from the shared start instead of staying fixed. 'fixed' keeps the pin.
            if self.p.get('humans_shared_want_signs') and self.is_cd_human():
                # The shared sought sign per good is written into the consumer's STANDING-WANT BLOCK by good index
                # (a row's good is read from the consumer's genes, so a row-level pin would land on the wrong good).
                # 'initial': once, here, into the CMA-ES starting mean; 'fixed': also into every candidate (step()).
                self.initialVec = self._write_shared_want_signs(np.array(list(float_vec), dtype=float))
                self.message, float_vec = self.float_vec_to_trade_plan(self.initialVec, mask)
                self.message, float_vec = self._apply_shared_want_signs(self.message, np.array(list(float_vec), dtype=float))
                self.initialVec = np.array(list(float_vec), dtype=float)

        self.float_vec = float_vec
        self.model.blackboard.append(self.message)

        # Variant-mode knobs (all default off so existing scenarios behave
        # identically). Set in scenario JSON under agent_parameters.RoleSim:
        #
        #   cma_diagonal: bool or int (default False)
        #     Enables sep-CMA-ES via pycma's CMA_diagonal option. True means
        #     diagonal C forever (no off-diagonal covariance learning, scales
        #     to high dim, treats coords independently — fits problems where
        #     dimensions are largely uncorrelated, e.g. cc1's mostly-
        #     categorical decode where there's little cross-dim structure to
        #     learn). int N means diagonal for first N iterations then full.
        #
        #   ipop: bool (default False)
        #     Enables IPOP-CMA-ES style restarts. After each generation,
        #     if es.stop() reports any stagnation/convergence condition AND
        #     fewer than ipop_max_restarts have been used, replace self.es
        #     with a fresh CMAEvolutionStrategy at doubled popsize and a
        #     freshly resampled initial mean. Breaks flat-fitness deadlocks
        #     where the current distribution has no productive samples and
        #     CMA-ES has no fitness gradient to escape it.
        #
        #   ipop_max_restarts: int (default 9)
        #     Bound on number of IPOP restarts. 9 is the canonical default.
        #
        self.cma_diagonal = self.parameters.get('cma_diagonal', False)
        self.ipop_enabled = bool(self.parameters.get('ipop', False))
        self.ipop_max_restarts = int(self.parameters.get('ipop_max_restarts', 9))
        self.ipop_restart_count = 0
        #   ONE board per candidate (her rulings of 2 and 3 Oct 2026): judging a
        #   candidate on several boards (steps_per_solution) was removed; the
        #   noise washes out as quickly without it.
        #   buyer_score_accumulation: 'max' (default) or 'sum'
        #     'sum' accumulates every settled purchase's score over the
        #     window — appetite: another good purchase always adds fitness.
        #     'max' keeps only the best purchase (original behavior).
        #   buyer_score_exponent: float (default 1.0)
        #     Exponent applied to each purchase score before accumulating.
        #     2.0 makes the value of quality convex, so higher scores are
        #     worth more than proportionally and quantity of junk cannot
        #     outweigh quality.
        self.buyer_score_accumulation = str(self.parameters.get('buyer_score_accumulation', 'max')).lower()
        self.buyer_score_exponent = float(self.parameters.get('buyer_score_exponent', 1.0))
        self._vs = vs  # save dim for IPOP restart resampling

        self.es = self._make_cma_strategy(self.initialVec, getattr(self, '_seed_sigma', None) or self.parameters['sigma'])
        self.solutions = self.es.ask()
        self.results = []
        self.next_solution = 0

        self.agiTokens = 0
        self.max_buyer_score = 0
        self.max_seller_score = 0

        print ("IN RoleSim init,"+self.b[self.unique_id]['label'])

    def _apply_natal_proportions(self, vec):
        # The CR sampler: each field family is seeded from the parent
        # proportions through the engine's own inverse-decode helpers,
        # uniform WITHIN the chosen interval, so chances match the
        # parent exactly while everything unmeasured stays random.
        prop = self.p.get('natal_proportions')
        if not prop:
            return vec
        import random as _r
        ss = self.p['sign_size']
        ts = self.trade_size()
        n_plans = self.n_plans()
        mix = prop.get('sign_mixture')
        if mix and mix.get('centers'):
            ci = _r.choices(range(len(mix['centers'])),
                            weights=mix['weights'])[0]
            jit = float(mix.get('jitter', 0.05))
            for d in range(min(ss, len(mix['centers'][ci]))):
                vec[d] = min(1.0, max(0.0,
                             _r.gauss(mix['centers'][ci][d], jit)))
        for i in range(n_plans):
            base = ss + i * ts
            tm = prop.get('plan_type_mix')
            if tm:
                choice = _r.choices(['buy', 'construct', 'sell'],
                                    weights=tm)[0]
                vec[base] = self.float_for_trade_type(choice)
            fm = prop.get('item_family_mix')
            if fm:
                wc = self.weights_for_level('ontology')
                fam = _r.choices(list(fm), weights=list(fm.values()))[0]
                if fam in wc:
                    # Her fix (2026-08-12): under the single-float
                    # encoding the family must be pinned by INTERVAL, so
                    # the number's finer digits still choose a product
                    # inside it. The old per-level encoder put its stop
                    # slice at the top of the interval while this decode
                    # puts it at the bottom, so a seeded family landed in
                    # the wrong one - asked for clusterers, produced
                    # preprocessors.
                    f = self.float_for_family(fam, draw=_r.random())
                    if f is None:
                        f = self.float_for_weighted_choice(
                            wc, fam,
                            previously_taken_space=self.p.get(
                                'chance_of_stop_codon', 0))
                    if f is not None:
                        vec[base + 1] = f
            # GENOME VERSION 4 OFFSETS (her design 2026-08-12): a plan is
            # type | item | midpoint | range | want pointer, and the sign
            # and the four test selectors moved into the shared want
            # blocks after the plans. These writes follow it.
            ph = prop.get('price_mid_hist')
            if ph and ph.get('probs'):
                b = _r.choices(range(len(ph['probs'])),
                               weights=ph['probs'])[0]
                lo, hi = ph['bins'][b], ph['bins'][b + 1]
                vec[base + 1 + self.p['item_size']] = _r.uniform(lo, hi)
            ns = prop.get('selector_none_share_natal')
            if ns is not None:
                none_share = float(self.p.get('selector_none_share', 0.5))
                # the selectors of the want block this plan points at
                _wb = self.want_block(vec[base + 1 + self.p['item_size'] + 2])
                sel0 = (self.want_block_offset()
                        + _wb * self.want_block_size() + ss)
                for k in range(4):
                    if _r.random() < float(ns):
                        vec[sel0 + k] = _r.uniform(0.0, none_share)
                    else:
                        vec[sel0 + k] = _r.uniform(none_share, 1.0)
        return vec

    def _make_cma_strategy(self, mean_vec, sigma, popsize_override=None):
        """Build a pycma CMAEvolutionStrategy with the agent's variant-mode
        flags. Used at __init__ and on every IPOP restart."""
        # Her reproducibility order (2026-08-10): the per-agent strategy
        # seed is a pure function of the CONFIG seed, the agent's
        # identity, and how many strategies this agent has built (IPOP
        # restarts advance it) — never a draw from the global stream,
        # whose position drifts between launches. Same config, same
        # agent, same build ordinal -> same strategy, every launch.
        self._cma_builds = getattr(self, '_cma_builds', 0) + 1
        seed = (int(self.p.get('seed', 0)) * 1000003
                + int(self.unique_id) * 7919
                + self._cma_builds * 104729) % (2**31 - 1) + 1
        # One audit line per strategy build so any two launches can be
        # diffed for seed identity (her reproducibility order).
        print('cma_strategy_seed agent=%s build=%s seed=%s'
              % (self.unique_id, self._cma_builds, seed))
        params = {'bounds': [0.0, 1.0], 'seed': seed, 'CMA_elitist': self.parameters['elitist']}
        # cma_popsize lets scenarios override CMA-ES per-generation population
        # size. Default is cma's `4 + int(3*log(d))` (~25 at d=1148), which
        # times 16 agents times the per-eval ML pipeline cost gives ~30 minute
        # generations on cc_3corpora_16agents. Setting cma_popsize=6 cuts the
        # generation time ~4x at the cost of slower convergence per generation.
        if popsize_override is not None:
            params['popsize'] = int(popsize_override)
        elif 'cma_popsize' in self.parameters:
            params['popsize'] = int(self.parameters['cma_popsize'])
        # AGENTS OF DIFFERENT CAPACITIES (her suggestion, 6 Oct 2026: "different settings of the cma es to make
        # different skill sets"): cma_minstd_by_learner, a list cycled by agent number, gives each learner the
        # smallest step its search may take in any gene (pycma's minstd), so some learners can hold a narrow slice
        # and others cannot. Humans are never limited. Absent, or 0 for a learner: no floor, exactly as before.
        _mins = self.parameters.get('cma_minstd_by_learner')
        if _mins and not self.is_cd_human():
            _m = float(_mins[int(self.unique_id) % len(_mins)] or 0)
            if _m > 0:
                params['minstd'] = _m
        # A CEILING ON EVERY GENE'S STEP (7 Oct 2026, 00:5x; flag-gated, off unless set; a debugging lever tried while
        # she sleeps, to be reported). Measured in the hard-keys ladder at 5,000 boards: the global step had fallen
        # to 0.036 (mean over learners) and yet a learner's section changed kind on half the ten-board steps (3-row
        # learners (106/232) unchanged, 16-row (93/232)), which the sign-formation model says no sign can mark. With
        # slices of 0.225 a step of 0.036 crosses a slice on about 1 board in 500, so the moving genes must be
        # stepping far more than the global step: the covariance inflates the directions selection does not care
        # about, and a neutral gene (a sign nobody reads yet, a row nobody buys from) ends up jittering most. cma_maxstd
        # bounds the per-gene step (pycma's maxstd) at a value such as the birth step 0.1: the temperature still
        # adapts freely below the ceiling (never locked, her rule); it only cannot heat a gene beyond where it began.
        _maxstd = self.parameters.get('cma_maxstd')
        if _maxstd:
            params['maxstd'] = float(_maxstd)
        if self.cma_diagonal:
            # pycma CMA_diagonal: True means diagonal C forever (sep-CMA-ES);
            # positive int N means diagonal for first N iterations then full.
            # Treat any truthy value other than int as True; pass ints through.
            if isinstance(self.cma_diagonal, bool):
                params['CMA_diagonal'] = True
            else:
                params['CMA_diagonal'] = int(self.cma_diagonal)
        return cma.CMAEvolutionStrategy(mean_vec, sigma, params)

    def _maybe_ipop_restart(self):
        """If IPOP is enabled and es.stop() reports any termination, replace
        self.es with a fresh strategy at doubled popsize and a resampled
        initial mean. Returns True if a restart happened."""
        if not self.ipop_enabled:
            return False
        if self.ipop_restart_count >= self.ipop_max_restarts:
            return False
        stop_dict = self.es.stop()
        if not stop_dict:
            return False
        old_popsize = self.es.popsize
        new_popsize = old_popsize * 2
        reasons = list(stop_dict.keys())
        print(f"IPOP restart {self.ipop_restart_count + 1}/{self.ipop_max_restarts} "
              f"for {self.b[self.unique_id]['label']}: popsize {old_popsize}→{new_popsize}, "
              f"reasons={reasons}")
        new_mean = np.random.uniform(low=0.0, high=1.0, size=(self._vs,))
        self.es = self._make_cma_strategy(new_mean, self.parameters['sigma'],
                                          popsize_override=new_popsize)
        self.ipop_restart_count += 1
        return True



    def _write_shared_want_signs(self, vec):
        """Write the configured shared sought sign per good (genome floats in [0,1]) into this consumer's
        standing-want blocks, one block per good in the order of humans_cd goods. Returns the vector."""
        table = self.p.get('humans_shared_want_signs_table') or {}
        goods = list(self.p.get('humans_cd', {}).get('goods', []))
        base = self.want_block_offset(); size = self.want_block_size(); n = int(self.p['sign_size'])
        for gi, g in enumerate(goods):
            sig = table.get(g)
            if not sig:
                continue
            off = base + gi * size
            for j in range(min(n, len(sig))):
                if off + j < len(vec):
                    vec[off + j] = float(sig[j])
        return vec

    def _apply_shared_want_signs(self, message, vec):
        """After a decode: write the shared sign per good into the vector's blocks AND into the decoded buy rows
        (a row's '_want' is its good's index), so the posted sought sign is exactly the convention whatever else
        the decode wrote near the blocks (7 Oct 23:5x: block 0's slot was partly overwritten by other pins)."""
        table = self.p.get('humans_shared_want_signs_table') or {}
        goods = list(self.p.get('humans_cd', {}).get('goods', []))
        vec = self._write_shared_want_signs(vec)
        full = bool(self.p.get('sign_space_full'))
        for t in message.get('trades', []):
            if t.get('type') != 'buy' or t.get('_want') is None:
                continue
            gi = int(t['_want'])
            if gi < 0 or gi >= len(goods) or not table.get(goods[gi]):
                continue
            sig = [float(x) for x in table[goods[gi]]]
            t['sign'] = [2.0 * x - 1.0 for x in sig] if full else list(sig)
        return message, vec

    def step(self):

        print("IN RoleSim step,"+self.b[self.unique_id]['label']+ " time "+ str(self.model.schedule.time))
        # print(self.b[self.unique_id]['label'] + ' in step')

        # append put the cumulative token reward as the reward for the last solution
        _fw = self.parameters['fitness_weights']
        if self.initial_trade_plan and self.p.get('human_fitness_weights'):
            # Authored (masked) human learners may carry their own
            # fitness weights: the demand side seeks satisfaction, the
            # supply side seeks tokens. Applies to ALL masked learners
            # as a class, never to named individuals.
            _fw = self.p['human_fitness_weights']
        result = self.agiTokens * _fw['agi_tokens']  \
                + self.max_buyer_score * _fw['buyer_score'] \
                + self.max_seller_score * _fw['seller_score']
        if self.p.get('humans_fitness_authored') and self.initial_trade_plan:
            # Her v14 law: the human's fitness IS the one authored
            # utility function over all its wants — weights and wallet
            # from the authored midpoints, exponents 0.7/0.3, the
            # aspiration thermostat inside — computed on this window's
            # acquisitions. Genome is conduct; this is judgment.
            result = self._authored_utility()

        bought_items = self.get_bought_items()
        self.model.print_reproduction_report_line(self,result, bought_items)

        # ONE board per candidate (her rulings of 2 and 3 Oct 2026): the
        # candidate's reward on this board is its fitness.
        self.results.append(result)

        # move a cursor that tells which solution you are on.
        # if there are none left, tell then ask, clearing reward buffer, resettinng cursor
        # take the next solutions and put the message on the blackboard
        self.next_solution += 1
        if self.next_solution >= len(self.solutions):
            # Log before tell() because tell() mutates sigma
            if hasattr(self.model, 'metrics'):
                self.model.metrics.log_generation(self, self.results, self.es.sigma)
            if self.parameters.get('cma_log_stds'):
                # the per-gene steps actually sampled (sigma times the covariance's diagonal scale), one line per
                # generation per agent, so the jitter diagnosis above can be read from the console log
                try:
                    _st = self.es.stds
                    _imax = int(max(range(len(_st)), key=lambda _i: _st[_i]))
                    print('cma_stds agent=%s gen=%s sigma=%.4f mean=%.4f max=%.4f at_gene=%d above_0.1=%d of %d snapped=%s'
                          % (self.unique_id, getattr(self.es, 'countiter', -1), float(self.es.sigma),
                             float(sum(_st) / len(_st)), float(_st[_imax]), _imax,
                             sum(1 for _s in _st if _s > 0.1), len(_st), getattr(self, '_last_snapped', '-')))
                except Exception as _e:  # noqa: BLE001
                    print('cma_stds agent=%s unavailable: %r' % (self.unique_id, _e))
            #Checkpoints as false enables the cma-es to accept the fixed solution seeds (partially fixed solutions)
            # that are set parts of the space to evolve around
            # pycma minimizes function_values; results are rewards (higher is
            # better), so negate them for tell(). metrics.csv keeps raw rewards.
            self.es.tell(self.solutions, [-r for r in self.results], check_points=False)
            self.results= []
            self.next_solution = 0
            # IPOP restart on stagnation if enabled (replaces self.es).
            # Must happen BEFORE the next ask() so the fresh strategy
            # produces the new generation.
            self._maybe_ipop_restart()
            self.solutions = self.es.ask()


        mask = None
        if self.initial_trade_plan:
            step = math.floor(self.model.schedule.time)
            initial_message = self.initial_trade_plan['initial_message'] \
                if 'initial_message' in self.initial_trade_plan else 0
            final_message = self.initial_trade_plan['final_message'] \
                if 'final_message' in self.initial_trade_plan else sys.maxsize
            message_period = self.initial_trade_plan['message_period'] \
                if 'message_period' in self.initial_trade_plan else 1

            if step >= initial_message and step <= final_message and step % message_period == 0:
                mask = copy.deepcopy(self.initial_trade_plan)
                if self.p.get('humans_fitness_authored'):
                    # Her v14 ruling (2026-08-09): pin the buy, the
                    # family, the corpus, the test, and the stop — the
                    # April-4 human — but the offered PRICE is conduct
                    # and stays learned, and the utility exponent,
                    # budget, and want weights are JUDGMENT: they live
                    # only in the authored fitness function, never as
                    # trade fields. The mask copy is stripped; the
                    # authored record (initial_trade_plan) keeps its
                    # midpoints for the fitness evaluator.
                    for _t in mask.get('trades', []):
                        if _t.get('type') == 'buy':
                            for _fk in ('midpoint', 'range', 'alpha',
                                        'budget'):
                                _t.pop(_fk, None)

        _cand = self.solutions[self.next_solution]
        if self.p.get('humans_shared_want_signs') == 'fixed' and self.is_cd_human():
            _cand = self._write_shared_want_signs(np.array(list(_cand), dtype=float))   # the buyer-side convention held fixed
        _marg = self.parameters.get('sticky_margin')
        if _marg:
            # THE DEAD-ZONE READING (7 Oct 2026, 01:3x; flag-gated, off unless set; a mechanism built while she
            # sleeps and run only as a test until her word). Measured: with the global step at 0.036 a learner's
            # section still changed kind on half the ten-board steps, because a candidate crosses a slice boundary
            # whenever its number lands past it, and with means spread inside slices of 0.225 that is about a quarter
            # of candidates per gene (section 17 of DEBUG_LADDER_20261006.md). Here a CHOICE gene (a row's type, an
            # item's levels, a pointer, the test selectors; never a sign or a price) is read at the agent's current
            # mean unless the candidate's number lies more than the margin from the mean, so small steps rarely change
            # the kind, a large step still can, and when the mean itself moves the choice moves with it (the Gray-
            # coding intent she named). The genome is NOT touched: the candidate the search asked for is the one it
            # is told about; only the reading of it for this board changes.
            try:
                _mean = np.asarray(self.es.mean, dtype=float)
                _mask = self.discrete_gene_mask()
                _cand = np.array(_cand, dtype=float, copy=True)
                _near = _mask & (np.abs(_cand - _mean) < float(_marg))
                self._last_snapped = int(_near.sum())
                _cand[_near] = _mean[_near]
            except Exception as _e:  # noqa: BLE001
                print('sticky_margin agent=%s unavailable: %r' % (self.unique_id, _e))
        new_message, float_vec = self.float_vec_to_trade_plan(_cand, mask)
        if self.p.get('humans_shared_want_signs') == 'fixed' and self.is_cd_human():
            new_message, float_vec = self._apply_shared_want_signs(new_message, np.array(list(float_vec), dtype=float))

        # STUCK SIGNS at display: the ascribed dimensions overwrite the
        # first k of the displayed sign. The genome's floats there are
        # moot BY DESIGN - ascription means you do not get a say -
        # disclosed as the arm's one inert block.
        _stuck = getattr(self, '_stuck_sign', None)
        if _stuck and isinstance(new_message.get('sign'), list):
            for _d in range(min(len(_stuck), len(new_message['sign']))):
                new_message['sign'][_d] = _stuck[_d]

        # BOUGHT SIGNS (her SDSS staging, arm B; approved 2026-08-11):
        # display intensity is a status good - each period the agent
        # PAYS sign_cost_rate times the sum of its displayed sign.
        # Loud, high signs cost; a low profile is nearly free. The
        # agent pays for its own act (autonomy intact); only earners
        # can afford to keep wearing the expensive sign, so purchased
        # status must be re-earned every period.
        _srate = float(self.p.get('sign_cost_rate', 0) or 0)
        if _srate > 0 and isinstance(new_message.get('sign'), list):
            self.agiTokens -= _srate * sum(
                float(x) for x in new_message['sign'])

        if self.p.get('one_sell_post'):
            # Her Decision 11 (2026-08-08): one sell posting per agent.
            # The post carries as many buys and constructs as the
            # genome likes, and the one posted product may sell to any
            # number of buyers. A second sell codon acts as a stop:
            # the posted plan is truncated there. The law governs the
            # BOARD only — the genome itself is never rewritten.
            _tr3 = new_message.get('trades', [])
            _seen_sell = False
            _cut3 = None
            for _i3, _t3 in enumerate(_tr3):
                _ty3 = _t3.get('type')
                if _ty3 == 'stop':
                    # Her correction (2026-08-09): stop works too — the
                    # posting ends at the first stop OR the second
                    # sell, whichever comes first. Keeps the upkeep
                    # honest: no rent charged on sells no buyer can
                    # read past a stop.
                    _cut3 = _i3 + 1
                    break
                if _ty3 == 'sell':
                    if _seen_sell:
                        _cut3 = _i3
                        break
                    _seen_sell = True
            if _cut3 is not None:
                new_message['trades'] = _tr3[:_cut3]


        self.agiTokens = 0
        self.max_buyer_score = 0
        self.max_seller_score = 0
        # HER FIXED MONEY (4 Oct 2026, "there should be a fixed amount of money, so that paying too much
        # for a good comes with an opportunity cost"): a Cobb-Douglas human starts every board with
        # humans_cd['money']; what it does not spend vanishes with the board (each board is itself).
        _cdm = (self.p.get('humans_cd') or {}).get('money')
        if _cdm is not None and self.is_cd_human():
            if self.p.get('price_list'):
                raise ValueError('humans_cd money needs learned prices: switch price_list off')
            self._money_left = float(_cdm)
        self.set_message(new_message)
        self.float_vec = float_vec

        # Persist the CMA-ES distribution mean each step so a transformer
        # can warm-start from the trained-up CMA-ES state. The mean is the
        # CMA-ES "best guess" — sampling from it with sigma=0 gives the
        # current canonical proposal. Core reads this file at agent init
        # when warmstart_from is in agent_parameters; the per-agent vector
        # initializes the (learnable) base_logit so the transformer starts
        # at CMA-ES's plateau and refines through REINFORCE.
        try:
            import json as _json
            outdir = self.parameters.get('warmstart_save_path') if self.parameters else None
            if outdir is None:
                outdir = self.p.get('output_path', '') + 'warmstart/'
            if not os.path.exists(outdir):
                os.makedirs(outdir, exist_ok=True)
            mean_vec = self.es.mean.tolist() if hasattr(self.es, 'mean') else self.float_vec
            p = os.path.join(outdir, f'agent_{self.unique_id}.json')
            now = int(self.model.schedule.time)
            try:  # a day-one save must never destroy a mature dump
                if now == 0 and os.path.exists(p):
                    prior = _json.load(open(p)).get('time', 0)
                    if prior > 0:
                        os.rename(p, f'{p}.t{prior}')
            except (OSError, ValueError):
                pass
            tmp = p + f'.tmp.{os.getpid()}'
            with open(tmp, 'w') as f:
                _json.dump({'mean': list(mean_vec), 'sigma': float(self.es.sigma),
                            'time': now}, f)
            os.replace(tmp, p)
        except Exception:
            pass  # don't crash sim on warmstart-save failure


    def _authored_utility(self):
        """THE HUMAN UTILITY (her 1a, accepted 2026-10-02): U = sum over the
        human's wants of q_w^gamma, q_w = the score of the good accepted for
        want w this board (re-scored by the authored tests), 0 if none.
        gamma = utility_gamma (2, hers). No aspiration (her 1 Oct: competing
        with the past made things short-lived), no budget, no price term
        (the price is learned, or listed). Judgment only."""
        acq = getattr(self, '_window_acq', [])
        self._window_acq = []
        if self.p.get('humans_cd') and self.is_cd_human():
            # HER COBB-DOUGLAS UTILITY (4 Oct 2026): every good wanted, equal fixed weights;
            # a delivery's score stands in for a quantity and deliveries add up; eps softens a
            # missing good (0.3 by the ladder mathematics).
            cd = self.p['humans_cd']
            goods = cd['goods']
            eps = float(cd.get('eps', 0.3))
            x = {g: 0.0 for g in goods}
            for rec in acq:
                if len(rec) > 3 and rec[3] in x and rec[1]:
                    x[rec[3]] += max(0.0, min(1.0, float(rec[1])))
            # Her 4 Oct addition ("the utility function should hold the informtion tht prices doo now"):
            # with humans_cd['weights'], each good's exponent is its weight over the weights' total. With
            # the old list prices as weights, a Cobb-Douglas buyer under a fixed budget spends each good's
            # weight as its share of its money, so the list's spending is reproduced. No weights: equal.
            wts = cd.get('weights')
            tot_w = float(sum(float(wts[g]) for g in goods)) if wts else 0.0
            u = 1.0
            for g in goods:
                u *= (eps + x[g]) ** ((float(wts[g]) / tot_w) if wts else (1.0 / len(goods)))
            # MONEY KEPT COUNTS (her yes, 6 Oct 2026, to her diagnosis: "it comes from being free to use the budget
            # up until the 100 is spent, becasue until that number there is no opportunity cost"): what the human
            # does not spend of the board's money counts in this same board's score at money_value per unit, so
            # every token it pays costs it something and the most it will pay for a delivery is the utility the
            # delivery adds over money_value (PRICES_MATH_20261006.md). Off unless humans_cd['money_value'] is set.
            mv = cd.get('money_value')
            if mv:
                u += float(mv) * max(0.0, float(getattr(self, '_money_left', 0.0) or 0.0))
            return u
        wants = [t for t in self.initial_trade_plan.get('trades', [])
                 if t.get('type') == 'buy']
        if not wants:
            return 0.0
        gamma = float(self.p.get('utility_gamma', 2.0))
        best = {}
        for _rec in acq:
            k, s = _rec[0], _rec[1]
            if s and s > best.get(k, 0.0):
                best[k] = float(s)
        tot = 0.0
        for w in wants:
            k = stop_cut(str(w.get('item', '')))
            q = max(0.0, min(1.0, best.get(k, 0.0)))
            tot += q ** gamma
        return tot

    def buyer_score_notification(self, score, tradenum):
        #print(self.b[self.unique_id]['label'] + ' wealth changes by ' + agiTokens + ' because of trade ' + str(tradenum))
        # A malformed chain evaluation (e.g. a mis-shaped arity-2 tree) can
        # deliver None or a non-scalar here; that is not a score. Ignore it,
        # consistent with malformed-chain = no reward everywhere else.
        if not isinstance(score, (int, float)):
            return
        if self.buyer_score_accumulation == 'sum':
            self.max_buyer_score += score ** self.buyer_score_exponent
        elif self.max_buyer_score < score:
            self.max_buyer_score = score

    def seller_score_notification(self, score, tradenum):
        #print(self.b[self.unique_id]['label'] + ' wealth changes by ' + agiTokens + ' because of trade ' + str(tradenum))
        if not isinstance(score, (int, float)):
            return
        if self.max_seller_score < score:
            self.max_seller_score = score

    def payment_notification(self, agiTokens, tradenum):
        #print(self.b[self.unique_id]['label'] + ' wealth changes by ' + agiTokens + ' because of trade ' + str(tradenum))
        self.agiTokens += agiTokens