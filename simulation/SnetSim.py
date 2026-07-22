import json
import os
import pickle
import sys
import re
from collections import OrderedDict
import copy

from boltons.cacheutils import LRU
from boltons.cacheutils import cachedmethod
from mesa import Model

from simulation.Registry import registry
from simulation import Registry as PortRegistry
from simulation._mesa_compat import StagedActivation
from simulation.instrumentation import MetricsLogger
from simulation.pickleThis import pickleThis
from simulation.Exogenous import Exogenous
from simulation.SISTER import SISTER
from simulation.GCoN import GCoN


class _NullCache:
    """Cache ablation: always misses, never stores. Used when SNETSIM_NO_CACHE=1.
    Exposes hit_count / miss_count so instrumentation still works."""
    def __init__(self):
        self.hit_count = 0
        self.miss_count = 0

    def __contains__(self, key):
        return False

    def __getitem__(self, key):
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.miss_count += 1  # track total evals

    def get(self, key, default=None):
        self.miss_count += 1
        return default

    def __len__(self):
        return 0


class SnetSim(Model):
    def __init__(self, study_path='study.json'):
        # Get data from config file
        with open(study_path) as json_file:
            config = json.load(json_file, object_pairs_hook=OrderedDict)
        self.parameters = config['parameters']
        seed = self.parameters['seed']
        super().__init__(seed=seed)
        # Seed all the RNGs that contribute to per-run variance:
        # Mesa's seed (above) covers Mesa's internal RNG, but downstream
        # libraries (torch, numpy, python random) maintain their own state.
        # Without explicit seeding, two runs with the same scenario seed
        # produce different transformer policy initializations and different
        # REINFORCE exploration noise, leading to the 46pp within-config
        # capture-rate variance observed in earlier IRA experiments.
        memcap = os.environ.get('SNETSIM_EVAL_MEMCAP_GB')
        if memcap:
            import resource
            cap = int(float(memcap) * 2**30)
            # RLIMIT_DATA (not AS): AS counts virtual reservations, which
            # numpy/torch exhaust at import; DATA covers brk + private mmap
            # on Linux >= 4.7 — i.e., actual array allocations.
            resource.setrlimit(resource.RLIMIT_DATA, (cap, cap))
        import random as _random
        _random.seed(seed)
        try:
            import numpy as _np
            _np.random.seed(seed)
        except Exception:
            pass
        try:
            import torch as _torch
            _torch.manual_seed(seed)
            if hasattr(_torch, 'cuda') and _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        self.blackboard = config['blackboard']
        self.ontology = config['ontology']

        # Copy config file to output folder
        outpath = config['parameters']['output_path']
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        filename = outpath + study_path
        pretty = json.dumps(config, indent=2, separators=(',', ':'))
        with open(filename, 'w') as outfile:
            outfile.write(pretty)

        # Initialize class attributes
        self.gepResult = None
        self.registry = registry
        self.reproduction_report = self.reproduction_report()
        self.emergent_functions = OrderedDict()
        self.emergent_functions_arity = OrderedDict()
        self.emergent_functions_call_number = 0
        self.stochastic_pattern = re.compile(r'_stochastic\d+')
        self.prefix_pattern = re.compile(r'^f\d+_')

        # Pickling parameters
        pickle_config_path = config['parameters']['output_path'] + 'pickles/' + 'index.p'
        if pickle_config_path and os.path.exists(pickle_config_path):
            try:
                with open(pickle_config_path, 'rb') as cachehandle:
                    pickle_config = pickle.load(cachehandle)
            except (EOFError, pickle.UnpicklingError):
                # Legacy index corrupted by concurrent writers in a shared
                # pickles namespace — it is vestigial (content-addressed
                # cache ignores it); start fresh.
                pickle_config = OrderedDict([("count", 0), ("pickles", OrderedDict())])
        else:
            pickle_config = OrderedDict([("count", 0), ("pickles", OrderedDict())])
        self.pickle_count = pickle_config['count']  # contains the next number for the pickle file
        self.pickles = pickle_config['pickles']

        self.resultTuple = ()
        # (usage-reweighting, earnings ledger, and imitation support
        # removed 2026-07-19: unauthorized mechanisms — Debbie's ruling.)
        # self.cache = LRU(max_size = 512)
        if os.environ.get("SNETSIM_NO_CACHE"):
            self.cache = _NullCache()  # ablation: every eval is a miss, nothing stored
        else:
            self.cache = LRU()
        self.metrics = MetricsLogger(self.parameters['output_path'], self)

        # Buyers gather offers by ranking those who offer to sell the product that have a price overlap.
        # Call `choose_partners` several times to ensure that the supply chain has a
        # chance to be settled in multiple trades, or offer networks have a chance to be filled.
        # In `step`, the agent has a chance to put out a new message given the knowledge of purchases
        # made on the last round.
        stage_list = ['step', 'gather_offers',
                      'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners',
                      'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners',
                      'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners',
                      'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners',
                      ]

        self.schedule = StagedActivation(self, stage_list=stage_list, shuffle=True, shuffle_between_stages=True)

        # Create initial agents as requested in `blackboard agents`
        initial_blackboard = copy.deepcopy(self.blackboard)
        self.blackboard = []
        agent_count = 0
        for i, message in enumerate(initial_blackboard):
            if message['type'] in self.parameters['agent_parameters']:
                agent_parameters = self.parameters['agent_parameters'][message['type']]
            else:
                agent_parameters = None
            # A blackboard entry may carry its own agent_parameters dict,
            # merged over the type-level defaults, so roles can differ in
            # utility (fitness_weights) or any other learner parameter.
            if 'agent_parameters' in message:
                agent_parameters = {**(agent_parameters or {}), **message['agent_parameters']}
            # Create as many copies of agent `i` as requested
            for _ in range(self.parameters['blackboard_agents'][i]):
                a = globals()[message['type']](agent_count, self, message, agent_parameters)
                self.schedule.add(a)
                agent_count += 1

        # Create random agents
        for agent_type, n in self.parameters['random_agents'].items():
            if agent_type in self.parameters['agent_parameters']:
                agent_parameters = self.parameters['agent_parameters'][agent_type]
            else:
                agent_parameters = None
            for i in range(n):
                a = globals()[agent_type](agent_count, self, None, agent_parameters)
                self.schedule.add(a)
                agent_count += 1

        print("Initialized SnetSim instance!")

    def remove_suffix(self, func_name):
        cut_tuple = func_name
        if func_name:
            stochastic_suffix = self.stochastic_pattern.search(func_name)
            if stochastic_suffix:
                stochastic_suffix = stochastic_suffix.group()
                cut_tuple = func_name[:-len(stochastic_suffix)]
        return cut_tuple

    def _stochastic_rng_sandbox(self, tuple_key):
        """Context-keyed seeding for stochastic chain evaluations.

        Gated on parameters['context_keyed_stochastic'] (default False =
        the historical behavior, byte-identical, protecting all runs
        launched before this feature). When ON and the chain root
        carries a _stochasticN suffix, the evaluation runs inside a
        private RNG sandbox seeded from hash(chain key + scenario seed):
        the draw becomes a pure function of (chain, copy index, seed) —
        same frozen-draw-per-copy semantics as the 2018 stochastic
        pickles — and the GLOBAL random stream is provably untouched
        whether the evaluation is a cache hit or a miss. That is what
        makes replay after a crash a true resumption instead of a
        silent fork (a cache hit skips the draws the original run
        consumed; without the sandbox the global stream offsets and the
        trajectory diverges from there).
        """
        import contextlib
        import hashlib
        import random as _random

        @contextlib.contextmanager
        def sandbox():
            if not self.parameters.get('context_keyed_stochastic', False):
                yield
                return
            if not (tuple_key and tuple_key[0] and
                    self.stochastic_pattern.search(tuple_key[0])):
                yield
                return
            import numpy as _np
            seed_material = f"{tuple_key}|{self.parameters.get('seed')}"
            seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:12], 16)
            py_state = _random.getstate()
            np_state = _np.random.get_state()
            _random.seed(seed)
            _np.random.seed(seed % (2**32))
            try:
                yield
            finally:
                _random.setstate(py_state)
                _np.random.set_state(np_state)
        return sandbox()

    @cachedmethod('cache')
    @pickleThis
    def memoise_pickle(self, tuple_key):
        # NOTE on swallowed exceptions: the modernized stack (numpy 1.20+,
        # gensim 4, sklearn 1.x) raises ValueError / AttributeError where
        # 2018-vintage libs accepted inputs silently. Keeping return-None on
        # exception preserves the framework's "test failed → score 0" semantics,
        # but we now persist each caught exception to
        # `<output_path>caught_exceptions.log` with traceback + function name +
        # arg types so the failures are diagnosable instead of invisible.
        result = None
        cut_tuple = None

        try:
            cut_tuple = self.remove_suffix(tuple_key[0])
            with self._stochastic_rng_sandbox(tuple_key):
                if len(self.resultTuple) and cut_tuple:
                    result = self.registry[cut_tuple](*self.resultTuple)()
                else:
                    result = self.registry[cut_tuple]()
        except (IOError, ValueError, AttributeError, TypeError,
                RuntimeError, IndexError, KeyError, MemoryError) as e:
            # MemoryError joined 2026-07-06: with SNETSIM_EVAL_MEMCAP set,
            # a product too expensive to build fails like any other broken
            # chain (score None) instead of taking the whole market down —
            # a 13.6GB composed anomaly-detector chain nearly exhausted
            # swap in the newcomer arm.
            self._log_caught_exception(cut_tuple or tuple_key[0], e)
        except Exception:
            # Re-raise unexpected exceptions so they're not silently masked.
            self._log_caught_exception(cut_tuple or tuple_key[0], sys.exc_info()[1])
            raise

        return result

    def _log_caught_exception(self, func_name, exc):
        """Append one structured record to caught_exceptions.log so the
        ValueError/TypeError/etc. that memoise_pickle swallows are visible
        post-hoc. File location: <output_path>caught_exceptions.log."""
        import traceback
        try:
            log_path = self.parameters['output_path'] + 'caught_exceptions.log'
            arg_types = ', '.join(type(a).__name__ for a in (self.resultTuple or ()))
            tb_text = traceback.format_exc()
            time_str = getattr(self.schedule, 'time', '?') if hasattr(self, 'schedule') else '?'
            with open(log_path, 'a') as f:
                f.write(
                    f"=== time={time_str} func={func_name} "
                    f"exc={type(exc).__name__}: {exc}\n"
                    f"arg_types=({arg_types})\n"
                    f"{tb_text}\n"
                )
        except Exception:
            # Don't let logging itself break the simulation.
            pass

    def remove_prefix(self, func_name):
        cut_tuple = func_name
        if func_name:
            call_number_prefix = self.prefix_pattern.search(func_name)
            if call_number_prefix:
                call_number_prefix = call_number_prefix.group()
                cut_tuple = func_name[len(call_number_prefix):]
        return cut_tuple

    def get_call_prefix(self, func_name):
        call_prefix = None
        if func_name:
            call_number_prefix = self.prefix_pattern.search(func_name)
            if call_number_prefix:
                call_prefix = call_number_prefix.group()
        return call_prefix

    def call_emergent_function(self, gep_result, root):
        print("SnetSim calling emergent function with root {0}  :  {1}".format(root, gep_result))
        self.gepResult = copy.deepcopy(gep_result)
        # Shared-scenario port hook (Debbie 2026-07-16): publish the chain
        # being evaluated so the port want-scorers can refuse a delivered
        # chain that wires a FOREIGN port corpus terminal inside itself.
        # Read only by Registry.PORT_BUYER_KEYED scorers during a market
        # audition; inert for every historical scenario.
        PortRegistry.PORT_CURRENT_CHAIN = self.gepResult
        try:
            func_tuple = self.call_memoise_pickle(root)
        finally:
            PortRegistry.PORT_CURRENT_CHAIN = None

        print("SnetSim called emergent function with root {0} with result {1}".format(root, func_tuple))
        return func_tuple

    def call_memoise_pickle(self, root):
        # right now, self.emergentfunctions looks like:
        # 		f1: a,b,f2,c,d
        # 		f2: e,d,f3,f,g
        # 		f3: h,i,j
        #
        # You should go through the original problem that you had in the modulargep.txt file in the project
        # directory its going to be a matter of creating a registry for a functionlist on the fly from the real
        # registry I think.
        result = None
        func_tuple = (None, None)
        if root:
            result_list = []
            func_list = []
            # argTuple = ()
            if root in self.gepResult:
                args = self.gepResult[root]
                # argTuple = tuple(args)
                for arg in args:
                    temp_func_tuple, temp_result = self.call_memoise_pickle(arg)
                    result_list.append(temp_result)
                    func_list.append(temp_func_tuple)
            carried_back = tuple(func_list)
            strip_prefix = self.remove_prefix(root)
            func_tuple = (strip_prefix, carried_back)
            # Shared-scenario port (Debbie 2026-07-16): buyer-keyed scorers
            # carry the auditing buyer's data terminal in their cache key —
            # a self-contained foreign chain drops the buyer's appended
            # terminal from the karva tree, so an identical tree audited by
            # two wants would otherwise return the first buyer's cached
            # score to the second. Key shape changes ONLY for roots listed
            # in Registry.PORT_BUYER_KEYED during an audition; every
            # historical scenario is byte-identical.
            if (strip_prefix in getattr(PortRegistry, 'PORT_BUYER_KEYED', ())
                    and getattr(PortRegistry, 'PORT_BUYER_DATA', None)):
                func_tuple = (strip_prefix, carried_back,
                              ('PORT_BUYER', PortRegistry.PORT_BUYER_DATA))

            self.resultTuple = tuple(result_list)  # You have to set a global to memoise and pickle correctly
            if strip_prefix is not None:
                result = self.memoise_pickle(func_tuple)

        return func_tuple, result

    def reproduction_report(self):
        path = self.parameters['output_path'] + 'reproduction_report.csv'
        file = open(path, "w")
        file.write("time;agent;label;utility;agi_tokens;buyer_score;seller_score;sign_displayed;bought_items\n")

        return file

    def print_reproduction_report_line(self, agent, utility, bought_items):
        a = self.schedule.time
        b = agent.unique_id
        c = agent.message['label']
        d = utility
        e = agent.agiTokens
        f = agent.max_buyer_score
        g = agent.max_seller_score
        h = agent.message['sign']
        i = bought_items

        self.reproduction_report.write("{0};{1};{2};{3};{4};{5};{6};{7};{8}\n".format(a, b, c, d, e, f, g, h, i))
        self.reproduction_report.flush()

    def print_logs(self):
        log_path = self.parameters['output_path'] + "logs/"
        filename = log_path + "log" + str(self.schedule.time) + ".txt"
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        # `default=str` handles numpy scalars/arrays/torch tensors that leak
        # into the blackboard via float-vec decoding — they serialize as their
        # printed form rather than crash. Blackboard logs are for inspection,
        # not for round-tripping, so this is fine.
        def _default(o):
            try:
                return o.tolist() if hasattr(o, 'tolist') else str(o)
            except Exception:
                return str(o)
        pretty = json.dumps(self.blackboard, indent=2, separators=(',', ':'), default=_default)
        with open(filename, 'w') as outfile:
            outfile.write(pretty)
            # json.dump(self.blackboard, outfile)

        pickle_path = self.parameters['output_path'] + 'pickles/'
        pickle_config_path = pickle_path + 'index.p'
        pickle_config = OrderedDict([('count', self.pickle_count), ('pickles', self.pickles)])

        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        with open(pickle_config_path, 'wb') as outfile:
            pickle.dump(pickle_config, outfile)

    def visualize(self):
        # todo: visualize changes in price and test score and relative wealth
        pass

    def step(self):
        """Advance the model by one step."""
        print("IN SnetSim step, time " + str(self.schedule.time))
        self.print_logs()
        # Snapshot (sign, item) pairs into MI sliding window before step
        if hasattr(self, 'metrics'):
            self.metrics.collect_pairs()
        # self.visualize() after learning agents are implemented
        self.schedule.step()

    def go(self):
        for i in range(self.parameters['max_iterations']):
            print("iteration " + str(i))
            self.step()


def main():
    snetsim = SnetSim(sys.argv[1]) if len(sys.argv) > 1 else SnetSim()
    snetsim.go()


if __name__ == '__main__':
    main()
