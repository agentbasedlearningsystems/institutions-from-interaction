import numpy as np
import random
import cma
import json
import copy
import math
import sys
import os

from simulation.SnetAgent import SnetAgent


class SISTER(SnetAgent):
    def __init__(self, unique_id, model, message, parameters):
        super().__init__(unique_id, model, message, parameters)

        # SISTER uses CMA-ES on float vectors, that are then converted
        # to trade plans.  If an initial message is put on the board,
        # this is converted to a floatvec, and is used to initialize cma-es

        vs = self.vector_size()
        self.initialVec = np.random.uniform(low=0.0, high=1.0, size=(vs,))
        if not message:
            self.message, float_vec = self.float_vec_to_trade_plan(self.initialVec)
            self.initial_trade_plan = None
        else:

            self.initial_trade_plan = copy.deepcopy(self.message)
            mask = copy.deepcopy(self.initial_trade_plan)
            self.message, float_vec = self.float_vec_to_trade_plan(self.initialVec, mask)

        self.float_vec = float_vec
        self.model.blackboard.append(self.message)

        # Variant-mode knobs (all default off so existing scenarios behave
        # identically). Set in scenario JSON under agent_parameters.SISTER:
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
        #   genotype_mutation_rate: float in [0, 1] (default 0.0)
        #     Per-coordinate GA-style mutation applied to self.solutions
        #     RIGHT AFTER es.ask() and BEFORE the solutions are used / passed
        #     to es.tell(). Each coordinate of each solution is independently
        #     replaced (per genotype_mutation_kind) with probability
        #     genotype_mutation_rate. Unlike the eps decoder hook in
        #     SnetAgent.choose_weighted (which mutates the decoded category
        #     only — phenotype mutation, doesn't update the float CMA-ES
        #     learns from), this mutates the float-vec the optimizer sees:
        #     the mutated solution IS what tell() receives, so the mutation
        #     becomes a heritable variation that CMA-ES can pull `m` toward
        #     when it produces fitness. This is the per-gene mutation that
        #     a GA would have natively; it gives CMA-ES the same exploration
        #     property in flat-fitness regions.
        #
        #   genotype_mutation_kind: 'uniform' or 'gaussian' (default 'uniform')
        #     'uniform': replace coord with uniform[0,1] draw.
        #     'gaussian': add N(0, genotype_mutation_sigma) noise and clamp to [0,1].
        #
        #   genotype_mutation_sigma: float (default 0.1) — only used with 'gaussian'.
        self.cma_diagonal = self.parameters.get('cma_diagonal', False)
        self.ipop_enabled = bool(self.parameters.get('ipop', False))
        self.ipop_max_restarts = int(self.parameters.get('ipop_max_restarts', 9))
        self.ipop_restart_count = 0
        self.genotype_mutation_rate = float(self.parameters.get('genotype_mutation_rate', 0.0))
        self.genotype_mutation_kind = str(self.parameters.get('genotype_mutation_kind', 'uniform')).lower()
        self.genotype_mutation_sigma = float(self.parameters.get('genotype_mutation_sigma', 0.1))
        #   steps_per_solution: int (default 1)
        #     Reward-sparsity remedy. Each CMA-ES candidate stays posted for
        #     this many consecutive market steps and its fitness is the
        #     reward summed over the window, giving each plan more chances
        #     to settle trades before it is scored. All agents keep adapting
        #     simultaneously at every step; no partner is held still.
        self.steps_per_solution = int(self.parameters.get('steps_per_solution', 1))
        self._window_reward = 0.0
        self._window_steps = 0
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

        self.es = self._make_cma_strategy(self.initialVec, self.parameters['sigma'])
        self.solutions = self.es.ask()
        if self.genotype_mutation_rate > 0:
            self.solutions = self._mutate_solutions(self.solutions)
        self.results = []
        self.next_solution = 0

        self.agiTokens = 0
        self.max_buyer_score = 0
        self.max_seller_score = 0

        print ("IN SISTER init,"+self.b[self.unique_id]['label'])

    def _make_cma_strategy(self, mean_vec, sigma, popsize_override=None):
        """Build a pycma CMAEvolutionStrategy with the agent's variant-mode
        flags. Used at __init__ and on every IPOP restart."""
        seed = random.randint(1, 1000000)
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
        if self.cma_diagonal:
            # pycma CMA_diagonal: True means diagonal C forever (sep-CMA-ES);
            # positive int N means diagonal for first N iterations then full.
            # Treat any truthy value other than int as True; pass ints through.
            if isinstance(self.cma_diagonal, bool):
                params['CMA_diagonal'] = True
            else:
                params['CMA_diagonal'] = int(self.cma_diagonal)
        return cma.CMAEvolutionStrategy(mean_vec, sigma, params)

    def _mutate_solutions(self, solutions):
        """Per-coordinate independent mutation of CMA-ES proposals. Each
        coordinate of each solution has probability genotype_mutation_rate
        of being mutated (uniform replacement or Gaussian additive noise).
        Returns a new list of np.ndarrays with the same shape.

        Mutating here — between es.ask() and self.float_vec_to_trade_plan —
        means the mutated genotype is what tell() will later see, so CMA-ES
        learns from heritable variation. This is the missing per-gene
        mutation operator that the eps decoder hook does NOT provide
        (eps mutates only the decoded category, leaving float_vec untouched).
        """
        if self.genotype_mutation_rate <= 0:
            return solutions
        out = []
        for sol in solutions:
            sol = np.asarray(sol, dtype=float).copy()
            mask = np.random.uniform(0.0, 1.0, size=sol.shape) < self.genotype_mutation_rate
            if not mask.any():
                out.append(sol)
                continue
            if self.genotype_mutation_kind == 'gaussian':
                noise = np.random.normal(0.0, self.genotype_mutation_sigma, size=sol.shape)
                sol[mask] = np.clip(sol[mask] + noise[mask], 0.0, 1.0)
            else:  # 'uniform'
                sol[mask] = np.random.uniform(0.0, 1.0, size=mask.sum())
            out.append(sol)
        return out

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



    def step(self):

        print("IN SISTER step,"+self.b[self.unique_id]['label']+ " time "+ str(self.model.schedule.time))
        # print(self.b[self.unique_id]['label'] + ' in step')

        # append put the cumulative token reward as the reward for the last solution
        result = self.agiTokens * self.parameters['fitness_weights']['agi_tokens']  \
                + self.max_buyer_score * self.parameters['fitness_weights']['buyer_score'] \
                + self.max_seller_score * self.parameters['fitness_weights']['seller_score']

        bought_items = self.get_bought_items()
        self.model.print_reproduction_report_line(self,result, bought_items)

        # Accumulate the current candidate's reward over its evaluation
        # window; only score it (and move on) when the window is complete.
        self._window_reward += result
        self._window_steps += 1
        if self._window_steps < self.steps_per_solution:
            self.agiTokens = 0
            self.max_buyer_score = 0
            self.max_seller_score = 0
            return
        self.results.append(self._window_reward)
        self._window_reward = 0.0
        self._window_steps = 0


        # move a cursor that tells which solution you are on.
        # if there are none left, tell then ask, clearing reward buffer, resettinng cursor
        # take the next solutions and put the message on the blackboard
        self.next_solution += 1
        if self.next_solution >= len(self.solutions):
            # Log before tell() because tell() mutates sigma
            if hasattr(self.model, 'metrics'):
                self.model.metrics.log_generation(self, self.results, self.es.sigma)
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
            # Per-gene mutation applied to the freshly-sampled solutions
            # so that tell() next generation sees the mutated genotypes.
            if self.genotype_mutation_rate > 0:
                self.solutions = self._mutate_solutions(self.solutions)


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

        new_message, float_vec = self.float_vec_to_trade_plan(self.solutions[self.next_solution],mask)



        self.agiTokens = 0
        self.max_buyer_score = 0
        self.max_seller_score = 0
        self.set_message(new_message)
        self.float_vec = float_vec

        # Persist the CMA-ES distribution mean each step so a transformer
        # can warm-start from the trained-up CMA-ES state. The mean is the
        # CMA-ES "best guess" — sampling from it with sigma=0 gives the
        # current canonical proposal. GCoN reads this file at agent init
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
            with open(os.path.join(outdir, f'agent_{self.unique_id}.json'), 'w') as f:
                _json.dump({'mean': list(mean_vec), 'sigma': float(self.es.sigma),
                            'time': int(self.model.schedule.time)}, f)
        except Exception:
            pass  # don't crash sim on warmstart-save failure


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