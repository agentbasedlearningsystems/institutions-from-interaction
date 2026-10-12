import json
import os
import pickle
import random
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

def preserve_existing(path):
    """Move an existing non-empty file aside rather than truncating it
    (rename is a directory op - works even on a full disk)."""
    try:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            stamp = int(os.path.getmtime(path))
            alt = '%s.prev%d' % (path, stamp)
            if not os.path.exists(alt):
                os.rename(path, alt)
    except OSError:
        pass

from simulation.instrumentation import MetricsLogger
from simulation.MarketAgent import stop_cut
from simulation.pickleThis import pickleThis
from simulation.Exogenous import Exogenous
from simulation.RoleSim import RoleSim
from simulation.Core import Core
from simulation.LLMAgent import LLMAgent   # the language-model brain (7 Oct 2026): same world, different agent


class _NullCache:
    """Cache ablation: always misses, never stores. Used when MARKETSIM_NO_CACHE=1.
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





class MarketSim(Model):
    def __init__(self, study_path='study.json'):
        # Get data from config file
        with open(study_path) as json_file:
            config = json.load(json_file, object_pairs_hook=OrderedDict)
        self.parameters = config['parameters']
        # CAPS AS CONFIG (her ruling 2026-10-02): the row cap, column cap and
        # delivery-memo cap are stated in the world's config, not hidden env.
        for _ck, _ev in (('vectorspace_max_rows', 'MARKETSIM_VECTORSPACE_MAX_ROWS'),
                         ('text_bridge_features', 'MARKETSIM_TEXT_BRIDGE_FEATURES')):
            if self.parameters.get(_ck) is not None:
                os.environ[_ev] = str(int(self.parameters[_ck]))
        seed = self.parameters['seed']
        if self.parameters.get('cipher_P') is not None:
            try:
                from simulation import cipher_world as _CWp
            except ImportError:
                import cipher_world as _CWp
            if int(self.parameters['cipher_P']) != int(_CWp.P):
                raise SystemExit('cipher_P in the config is %s but the module runs with P=%s: '
                                 'launch with CIPHER_P=%s' % (self.parameters['cipher_P'], _CWp.P,
                                                              self.parameters['cipher_P']))
            # the key settings are checked the same way (5 Oct 2026): a world whose environment does not match its
            # config would score another world's material
            if int(self.parameters.get('cipher_subst_k', 0) or 0) != int(_CWp.SUBST_K):
                raise SystemExit('cipher_subst_k in the config is %s but the module runs with CIPHER_SUBST_K=%s'
                                 % (self.parameters.get('cipher_subst_k', 0), _CWp.SUBST_K))
            if int(self.parameters.get('cipher_layers', 1) or 1) != int(getattr(_CWp, 'LAYERS', 1)):
                raise SystemExit('cipher_layers in the config is %s but the module runs with CIPHER_LAYERS=%s'
                                 % (self.parameters.get('cipher_layers', 1), getattr(_CWp, 'LAYERS', 1)))
            if bool(self.parameters.get('cipher_merge')) != bool(getattr(_CWp, 'MERGE', False)):
                raise SystemExit('cipher_merge in the config is %s but the module runs with CIPHER_MERGE=%s'
                                 % (self.parameters.get('cipher_merge'), getattr(_CWp, 'MERGE', False)))
            if bool(self.parameters.get('cipher_strict_judge')) != bool(os.environ.get('CIPHER_STRICT_JUDGE')):
                raise SystemExit('cipher_strict_judge in the config is %s but CIPHER_STRICT_JUDGE is %r'
                                 % (self.parameters.get('cipher_strict_judge'), os.environ.get('CIPHER_STRICT_JUDGE')))
            if bool(self.parameters.get('cipher_whole_aon')) != bool(os.environ.get('CIPHER_WHOLE_AON')):
                raise SystemExit('cipher_whole_aon in the config is %s but CIPHER_WHOLE_AON is %r'
                                 % (self.parameters.get('cipher_whole_aon'), os.environ.get('CIPHER_WHOLE_AON')))
            if bool(self.parameters.get('cipher_edge_free')) != bool(getattr(_CWp, 'EDGE_FREE', False)):
                raise SystemExit('cipher_edge_free in the config is %s but the module runs with CIPHER_EDGE_FREE=%s'
                                 % (self.parameters.get('cipher_edge_free'), getattr(_CWp, 'EDGE_FREE', False)))
            if int(self.parameters.get('cipher_keysets', 1) or 1) != int(getattr(_CWp, 'KEYSETS', 1)):
                raise SystemExit('cipher_keysets in the config is %s but the module runs with CIPHER_KEYSETS=%s'
                                 % (self.parameters.get('cipher_keysets', 1), getattr(_CWp, 'KEYSETS', 1)))
        # THE WHOLE'S BAR in the machine-learning world (her yes, 6 Oct 2026, decision 10): the config's ml_whole_bar
        # (corpus -> bar) must be what the registry runs with (ML_WHOLE_BAR), or the world refuses to start.
        _mwb = self.parameters.get('ml_whole_bar')
        _mwb_env = os.environ.get('ML_WHOLE_BAR')
        _mwb_str = ','.join('%s:%s' % (k, _mwb[k]) for k in sorted(_mwb)) if _mwb else None
        _env_norm = ','.join(sorted(x.strip() for x in _mwb_env.split(',') if x.strip())) if _mwb_env else None
        if (_mwb_str or None) != (_env_norm or None):
            raise SystemExit('ml_whole_bar in the config is %r but ML_WHOLE_BAR is %r' % (_mwb_str, _mwb_env))
        # THE DOWNSTREAM TEST'S CLASSIFIER (her word, 6 Oct 2026, decision 11): the config's ml_downstream_c must be what
        # the registry runs with (ML_DOWNSTREAM_C), or the world refuses to start.
        _mdc = self.parameters.get('ml_downstream_c')
        _mdc_env = os.environ.get('ML_DOWNSTREAM_C')
        if (None if _mdc is None else float(_mdc)) != (None if _mdc_env is None else float(_mdc_env)):
            raise SystemExit('ml_downstream_c in the config is %r but ML_DOWNSTREAM_C is %r' % (_mdc, _mdc_env))
        super().__init__(seed=seed)
        # Seed all the RNGs that contribute to per-run variance:
        # Mesa's seed (above) covers Mesa's internal RNG, but downstream
        # libraries (torch, numpy, python random) maintain their own state.
        # Without explicit seeding, two runs with the same scenario seed
        # produce different transformer policy initializations and different
        # REINFORCE exploration noise, leading to the 46pp within-config
        # capture-rate variance observed in earlier IRA experiments.
        # Her law: science-affecting values live in CONFIG, never in
        # ambient env. Config is the source of truth; the engine
        # propagates to the env names the deep layers read.
        for _ck, _ev in (('eval_memcap_gb', 'MARKETSIM_EVAL_MEMCAP_GB'),
                         ('vectorspace_max_rows', 'MARKETSIM_VECTORSPACE_MAX_ROWS'),
                         # solver_max_iter (her order 2026-08-12): the
                         # deterministic work bound for estimators whose
                         # iteration count is otherwise unlimited.
                         ('solver_max_iter', 'MARKETSIM_SOLVER_MAX_ITER'),
                         ('data_nrows', 'CORE_DATA_NROWS')):
            _cv = self.parameters.get(_ck)
            if _cv is not None:
                os.environ[_ev] = str(_cv)
        memcap = os.environ.get('MARKETSIM_EVAL_MEMCAP_GB')
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
        # a24 (her order 2026-08-12, "default"): the single-float ontology
        # encoding is now the default. When it is on, an ontology
        # reference IS one float - the leading part naming the family and
        # the finer parts the branches - so the layout sizes follow from
        # the encoding rather than being set beside it. Forcing them here
        # means no config can be half-converted: a stale item_size of 8
        # would otherwise leave seven dead floats in every trade block.
        if self.parameters.get('single_float_ontology', True):
            self.parameters['item_size'] = 1
            self.parameters['test_path_size'] = 1

        self.blackboard = config['blackboard']
        self.ontology = config['ontology']

        # Copy config file to output folder
        outpath = config['parameters']['output_path']
        # HER RULING (2026-08-20): "they are supposed to play from the
        # start in a way that reproduces the run, and that rushes
        # through cache to the same place. but yes they should do it in
        # another file." Replay-from-zero is the design; overwriting the
        # previous life's boards is not. Measured on ubuntu-32gb-nbg1-2:
        # log100 rewritten at 17:17 while log6000 still held content
        # written at 14:11 and log2000 content from 07:33 - ONE directory
        # carrying three runs, so every "last 20 boards" read on it
        # silently mixed lives. So: if this output directory already
        # holds board logs, the whole previous life is moved aside to
        # <name>_arch_<stamp> - the standing _arch_ convention for
        # previous runs - and the replay starts on a clean directory.
        # Stamped from the logs directory's own mtime, matching
        # preserve_existing's .prev<mtime> convention, so a stamp always
        # says WHEN that life last wrote.
        try:
            _pl = os.path.join(outpath, 'logs')
            if os.path.isdir(_pl) and any(
                    _f.startswith('log') and _f.endswith('.txt')
                    for _f in os.listdir(_pl)):
                _base = outpath.rstrip('/')
                _arch = '%s_arch_%d' % (_base, int(os.path.getmtime(_pl)))
                if not os.path.exists(_arch):
                    os.rename(_base, _arch)
                    print('ARCHIVED previous life to ' + _arch)
        except OSError:
            pass
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        filename = outpath + study_path
        pretty = json.dumps(config, indent=2, separators=(',', ':'))
        preserve_existing(filename)
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
        # HER RULING (2026-08-13, "first yes"): the shared cache index is NOT
        # loaded. It accumulates across the project's entire history rather
        # than the run's, so every launch inherited all of it - measured at
        # 1,246,309 entries and 0.88 GB resident, on top of which the run
        # then grew its own. That is why memory was stable when the cache was
        # young and is not now: a run's footprint had come to depend on how
        # much work the project had ever done.
        #
        # Nothing is lost. A cache hit is decided by whether the content-
        # addressed file exists on disk, never by this index, and the only
        # consumer of the mapping is the delivery record, which needs the
        # keys this run itself touches - and gets them as it touches them.
        self.pickle_count = 0
        self.pickles = OrderedDict()

        self.resultTuple = ()
        # (usage-reweighting, earnings ledger, and imitation support
        # removed 2026-07-19: unauthorized mechanisms — the curator's ruling.)
        # self.cache = LRU(max_size = 512)
        if os.environ.get("MARKETSIM_NO_CACHE"):
            self.cache = _NullCache()  # ablation: every eval is a miss, nothing stored
        elif os.environ.get("CORE_MEM_FIX"):
            self.cache = LRU(max_size=int(os.environ.get("CORE_MEM_FIX_LRU", "512")))
        else:
            self.cache = LRU()
        self.metrics = MetricsLogger(self.parameters['output_path'], self)

        # Buyers gather offers by ranking those who offer to sell the product that have a price overlap.
        # Call `choose_partners` several times to ensure that the supply chain has a
        # chance to be settled in multiple trades, or offer networks have a chance to be filled.
        # In `step`, the agent has a chance to put out a new message given the knowledge of purchases
        # made on the last round.
        # HER RULING (m5, 2026-08-21): the number of arranging rounds is a
        # CONFIG PARAMETER, not a literal list. Measured on the boards: a buy
        # walks its runnable pool one candidate per round, pools run to 40
        # (median 25, p95 35), and twenty rounds let only 13.6% of buys finish
        # their search. She set it to fifty. Late rounds are nearly free -
        # rounds 16-20 carried 5.3% of all transfers between them.
        _rounds = int(self.parameters.get('arranging_rounds', 20))
        stage_list = ['step', 'gather_offers'] + ['choose_partners'] * _rounds

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

        print("Initialized MarketSim instance!")

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

        UNCONDITIONAL AND UNIVERSAL as of 2026-08-11 (her laws:
        reproducibility is not an option, and every stochastic
        registry function is set deterministically in a wrapper).
        EVERY registry evaluation - not only stochastic-suffixed
        chains - runs in a private RNG sandbox: global python random
        and numpy random are context-seeded for the duration and
        restored after, so unpinned estimator randomness and any
        registry-internal shuffling are pure functions of the chain,
        identical across machines and cache states.

        THE SEED MATERIAL IS THE CHAIN, AND ONLY THE CHAIN (her
        decision a2, 2026-08-12: "drop the scenario seed from that
        text, seed the sandbox from the chain alone"). It previously
        read f"{tuple_key}|{scenario seed}", which made the same
        program on the same data compute one answer in society 318 and
        another in society 319. With one shared cache (decision a1)
        that meant whoever evaluated first wrote a value carrying an
        unrelated society's fingerprint, and every other society read
        it forever - first-writer-canonical doing load-bearing work
        where it should be a formality.

        Seeding from the chain alone makes a chain's value a property
        of the chain: same program, same data, same answer, on every
        machine, in every society, whether computed yesterday or never
        yet. The scenario seed keeps varying THE SOCIETY - births,
        genome initialization, the optimizer's sampling, the funnel
        draws, deaths, per-agent strategy seeds - which is what it was
        always for. What a program computes is technology, not
        society.

        A stochastic product still varies within a world: a chain root
        carrying a _stochasticN suffix is a DIFFERENT chain identity,
        so each copy draws differently - the same frozen-draw-per-copy
        semantics as the 2018 stochastic pickles. What changes is only
        that copy N agrees across societies.

        The GLOBAL random stream is provably untouched whether the
        evaluation is a cache hit or a miss. That is what makes replay
        after a crash a true resumption instead of a silent fork (a
        cache hit skips the draws the original run consumed; without
        the sandbox the global stream offsets and the trajectory
        diverges from there). Worlds recorded earlier keep their
        recorded trajectories as history, and entries already in the
        store keep their recorded values under first-writer-canonical:
        every stored value is still a legitimate evaluation of its
        chain, and a2 governs future first-writes.
        """
        import contextlib
        import hashlib
        import random as _random

        @contextlib.contextmanager
        def sandbox():
            import numpy as _np
            seed_material = f"{tuple_key}"
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

    def _eval_in_child(self, thunk, seconds):
        """Run thunk() in a forked child so a cap can stop work that the
        interpreter cannot interrupt (her order 2026-08-12, second
        precaution).

        A Python signal cannot end a computation inside a C library: the
        handler only runs between interpreter instructions, so a solver
        that never returns keeps the process for as long as it likes.
        Instance A sat inside one support-vector fit for nearly an hour
        with a 300-second cap nominally in force. The only bound that
        holds is a separate process the parent can kill.

        Returns (True, value) on completion, (False, None) if the cap
        fired or the child died. Fork is copy-on-write, so a large parent
        costs page tables rather than memory, and the child inherits the
        seeded RNG state, so what it computes is what the parent would
        have computed.

        DEFAULT OFF, and deliberately so: a clock-based verdict is a fact
        about this machine, not about the chain, and under
        first-writer-canonical it would be recorded for every society and
        machine forever. That is the thing a1 and a2 removed from the
        cache today. Turning this on needs her ruling on whether a
        timeout may be written to the shared store at all.
        """
        import os as _os
        import pickle as _pk
        import select as _sel
        import signal as _sig
        import time as _t

        r, w = _os.pipe()
        pid = _os.fork()
        if pid == 0:                                   # child
            try:
                _os.close(r)
                _data = _pk.dumps(thunk(), protocol=4)
                _os.write(w, len(_data).to_bytes(8, 'little'))
                _sent = 0
                while _sent < len(_data):
                    _sent += _os.write(w, _data[_sent:_sent + (1 << 20)])
                _os.close(w)
                _os._exit(0)
            except BaseException:
                try:
                    _os.close(w)
                except Exception:
                    pass
                _os._exit(3)
        _os.close(w)
        deadline = _t.monotonic() + float(seconds)
        buf = b''
        timed_out = False
        try:
            while True:
                left = deadline - _t.monotonic()
                if left <= 0:
                    timed_out = True
                    break
                ready, _, _ = _sel.select([r], [], [], left)
                if not ready:
                    timed_out = True
                    break
                chunk = _os.read(r, 1 << 20)
                if not chunk:
                    break
                buf += chunk
        finally:
            _os.close(r)
            if timed_out:
                try:
                    _os.kill(pid, _sig.SIGKILL)
                except OSError:
                    pass
            try:
                _os.waitpid(pid, 0)
            except OSError:
                pass
        if timed_out or len(buf) < 8:
            return False, None
        size = int.from_bytes(buf[:8], 'little')
        if len(buf) - 8 < size:
            return False, None
        try:
            return True, _pk.loads(buf[8:8 + size])
        except Exception:
            return False, None

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
                # THE FORCING FIX (the curator's order, 2026-08-12).
                # Registry entries come in two shapes. `curr(f)` stores
                # the curried GENERATOR: calling it binds arguments and
                # returns a thunk, and the thunk must be called once
                # more to force evaluation. Pre-applied entries such as
                # curries['data_freetext_csvColumn'](path) are already
                # thunks and force on their first call. The
                # zero-argument branch below assumed the second shape
                # only, so every `curr(f)` terminal that takes no
                # arguments - all seven data_vector_* synthetic
                # datasets - evaluated to the curried FUNCTION instead
                # of its data. Every parent that received it raised
                # TypeError, and first-writer-canonical then recorded
                # None for that chain permanently: zero of 2,570
                # deliveries in the wave 19 record used a synthetic
                # dataset, and the world's one two-input product threw
                # 55 times with arg_types=(function, function).
                # Forcing while the value is still callable serves both
                # shapes; no product in the registry returns a callable
                # as its result.
                if len(self.resultTuple) and cut_tuple:
                    result = self.registry[cut_tuple](*self.resultTuple)
                else:
                    result = self.registry[cut_tuple]()
                if callable(result):
                    result = result()
        except (IOError, ValueError, AttributeError, TypeError,
                RuntimeError, IndexError, KeyError, MemoryError) as e:
            # THE INTENDED SEMANTICS, RESTORED (2026-08-14): the NOTE above
            # promises "test failed -> score 0" - None on any caught
            # failure. But when the FORCE raised (a curried parent fed the
            # None of its failed child), `result` still held the unforced
            # curry assigned the line before, and that function object
            # became the chain's "value" - unpicklable, so never cached,
            # so recomputed forever: 91,737 times in the 148 iterations of
            # the stopped repro attempt, 87,633 in the twin. None restores
            # the documented semantics and lets the brokenness be cached
            # as the chain-fact it is.
            result = None
            # MemoryError joined 2026-07-06: with MARKETSIM_EVAL_MEMCAP set,
            # a product too expensive to build fails like any other broken
            # chain (score None) instead of taking the whole market down —
            # a 13.6GB composed anomaly-detector chain nearly exhausted
            # swap in the newcomer arm.
            # Her ruling 2026-08-11: chain-intrinsic failures ARE
            # cacheable facts (a broken chain is always broken);
            # only MACHINE-dependent failures (the memory cap here,
            # the time cap in pickleThis) must stay out of the
            # shared store - they describe the box, not the chain.
            # Her final ruling (2026-08-11): FIRST WRITER CANONICAL.
            # Every outcome, None included, persists to the shared
            # store - whoever evaluates a chain first determines its
            # recorded value for all machines and all time, because
            # reproducibility outranks machine-fairness. The caps
            # (memory, time) are part of the world's constitution,
            # set in config, identical everywhere.
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
        # BOUNDED (her order, 2026-08-12, shape left to me). Unbounded, this
        # file reached 227 MB and 340 MB in ninety minutes on the two v27
        # runs - 3.8 and 5.7 million lines, nearly all of them the same
        # handful of failures repeated. A log that large is not read, and it
        # competes with the runs for disk.
        #
        # The shape: a SIGNATURE is (function, exception type, message). The
        # first few occurrences of each signature are written in full, with
        # the traceback, because the first ones are what diagnosis needs.
        # After that the record is counted, not written, and a one-line
        # tally is emitted only when the count reaches a power of ten. So a
        # failure that happens five million times costs five full records
        # and six tally lines instead of five million tracebacks, and a
        # failure that happens once is still recorded in full. Nothing is
        # hidden: every signature keeps its exact count.
        _FULL_PER_SIGNATURE = 5
        try:
            if os.environ.get('MARKETSIM_WARM_CHILD'):
                return
            log_path = self.parameters['output_path'] + 'caught_exceptions.log'
            arg_types = ', '.join(type(a).__name__ for a in (self.resultTuple or ()))
            time_str = getattr(self.schedule, 'time', '?') if hasattr(self, 'schedule') else '?'
            if not hasattr(self, '_exc_signature_counts'):
                self._exc_signature_counts = {}
            sig = (func_name, type(exc).__name__, str(exc)[:200])
            n = self._exc_signature_counts.get(sig, 0) + 1
            self._exc_signature_counts[sig] = n
            if n <= _FULL_PER_SIGNATURE:
                with open(log_path, 'a') as f:
                    f.write(
                        f"=== time={time_str} func={func_name} "
                        f"exc={type(exc).__name__}: {exc}\n"
                        f"arg_types=({arg_types})\n"
                        f"{traceback.format_exc()}\n"
                    )
            elif n and (n & (n - 1) == 0 or str(n).strip('0') == '1'):
                # powers of ten (and of two, which costs nothing and makes
                # the early growth visible)
                with open(log_path, 'a') as f:
                    f.write(
                        f"--- tally time={time_str} func={func_name} "
                        f"exc={type(exc).__name__}: {str(exc)[:120]} "
                        f"-> {n} occurrences so far\n"
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
        if not os.environ.get('MARKETSIM_QUIET'):
            print("MarketSim calling emergent function with root {0}  :  {1}".format(root, gep_result))
        self.gepResult = copy.deepcopy(gep_result)
        # Shared-scenario port hook (the curator 2026-07-16): publish the chain
        # being evaluated so the port want-scorers can refuse a delivered
        # chain that wires a FOREIGN port corpus terminal inside itself.
        # Read only by Registry.PORT_BUYER_KEYED scorers during a market
        # audition; inert for every historical scenario.
        PortRegistry.PORT_CURRENT_CHAIN = self.gepResult
        try:
            func_tuple = self.call_memoise_pickle(root)
        finally:
            PortRegistry.PORT_CURRENT_CHAIN = None

        if not os.environ.get('MARKETSIM_QUIET'):
            print("MarketSim called emergent function with root {0} with result {1}".format(root, func_tuple))
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
            # Shared-scenario port (the curator 2026-07-16): buyer-keyed scorers
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
            if strip_prefix and str(strip_prefix).startswith(('decoder_', 'data_cipher_', 'test_decoder_')):
                # Cipher results carry the module's material salt (source hash and
                # position-class count): found 3 Oct, results cached under an older
                # module were being returned to later worlds.
                try:
                    from simulation import cipher_world as _CWsalt
                except ImportError:
                    import cipher_world as _CWsalt
                func_tuple = func_tuple + (('CIPHER', _CWsalt.cache_salt()),)

            self.resultTuple = tuple(result_list)  # You have to set a global to memoise and pickle correctly
            if strip_prefix is not None:
                result = self.memoise_pickle(func_tuple)

        return func_tuple, result

    def reproduction_report(self):
        path = self.parameters['output_path'] + 'reproduction_report.csv'
        preserve_existing(path)
        file = open(path, "w")
        file.write("time;agent;label;utility;agi_tokens;buyer_score;seller_score;sign_displayed;bought_items\n")

        return file

    def print_reproduction_report_line(self, agent, utility, bought_items):
        if os.environ.get('MARKETSIM_WARM_CHILD'):
            return  # warm children write no records
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
        # LIGHTER LOGS (her go 4 Oct 2026, "lengthen the run" / "i meant the log"): with log_every N > 1
        # the board log is written only on every Nth board, so long runs fit on disk. Default 1 = every board.
        _le = int(self.parameters.get('log_every', 1) or 1)
        if _le > 1 and int(round(float(self.schedule.time))) % _le != 0:
            return
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
        # Hidden law (the curator 2026-08-10: "make hidden the default and
        # take the flag out"): a buy's tests are never displayed on the
        # board. The in-memory blackboard keeps them (the buyer's own
        # settlement machinery reads them); the PERSISTED board - what
        # any market participant could see - carries item, sign, and
        # price band only.
        def _strip_tests(entry):
            if not isinstance(entry, dict):
                return entry
            e = dict(entry)
            if isinstance(e.get('trades'), list):
                e['trades'] = [{k: v for k, v in t.items() if k != 'tests'}
                               if isinstance(t, dict) else t
                               for t in e['trades']]
            return e
        # Board cleanup (her order 2026-08-17): slots that post nothing
        # - dead stubs and empty entries, no sign and no trades - are
        # not persisted. Offers reference sellers by in-memory
        # blackboard position, so each surviving entry carries that
        # position as 'board_index'; readers key on it via
        # scripts/composed_share.board_map, falling back to list
        # position for logs written before this change. The in-memory
        # blackboard is untouched - dead agents still hold their slots.
        _entries = []
        for _i, _e in enumerate(self.blackboard):
            if not isinstance(_e, dict) or not (_e.get('sign')
                                                or _e.get('trades')):
                continue
            # HER CORRECTION (2026-08-20): "all this should be in the
            # log; hidden meant it is not part of what the learner can
            # see." The hidden law governs the in-memory blackboard -
            # what agents can access - never the persisted record,
            # which only the instruments read. Tests (and their evolved
            # thresholds) persist in full.
            _se = dict(_e) if isinstance(_e, dict) else _e
            _se['board_index'] = _i
            _entries.append(_se)
        pretty = json.dumps(_entries,
                            indent=2, separators=(',', ':'), default=_default)
        preserve_existing(filename)
        with open(filename, 'w') as outfile:
            outfile.write(pretty)
            # json.dump(self.blackboard, outfile)

        # INDEX CADENCE (her decision a-round item, 2026-08-12: "do it
        # if it really helps"). Measured on the live index at 1,061,504
        # entries: 137.8 MB on disk, 3.48 seconds to serialize and a
        # 364 MB memory spike, EVERY iteration - about 39% of instance
        # A's wall clock, 11.6 hours over a 12,000-iteration run, and
        # the spike that met the memory cap and killed instance A at
        # iteration 1016. The index is vestigial for cache correctness
        # (filenames are content-addressed, so a lookup never consults
        # it); it is read once at startup to seed self.pickles, which
        # the in-memory mapping keeps current regardless. Writing it
        # every N iterations therefore risks nothing but the last N
        # iterations' entries in the file, and those are recoverable by
        # the next write. One config number, no env var.
        # NOT WRITTEN ANY MORE (her ruling 2026-08-13). Two reasons, and
        # either alone is sufficient.
        #
        # It is not read. The index was loaded once at startup to seed
        # self.pickles, and that load is gone: it made a run's memory depend
        # on the project's whole history rather than its own. A cache hit is
        # decided by whether the content-addressed file exists on disk.
        #
        # And it could no longer be written correctly. self.pickles is now
        # cleared at the end of every step, so a dump would contain one
        # step's entries and would REPLACE the 161 MB accumulated file with
        # them - turning a harmless leftover into an actively wrong one.
        #
        # The existing index.p is left on disk untouched. Nothing reads it;
        # deleting it is hers to decide, not a side effect of this change.
        # index_dump_every is now inert and stays in configs harmlessly.
        pass

    def visualize(self):
        # todo: visualize changes in price and test score and relative wealth
        pass

    def step(self):
        """Advance the model by one step."""
        print("IN MarketSim step, time " + str(self.schedule.time))
        self.print_logs()
        # Snapshot (sign, item) pairs into MI sliding window before step
        if hasattr(self, 'metrics'):
            self.metrics.collect_pairs()
        # self.visualize() after learning agents are implemented

        # BOARD-LEVEL SHARDED WARM (her cores order, 2026-08-18: "fix
        # them so that they use all the cores and restart them").
        # W children fork as copy-on-write snapshots of the live world
        # and speculatively run THIS board's step with a cloned RNG,
        # each computing only its hash-slice of the fresh chains into
        # the shared cache (atomic per-pid writes); record files are
        # barred child-side. The parent then restores the RNG and runs
        # the real step, replaying the frontier as cache hits. Results
        # are identical by cache transparency - the md5 identity gate
        # proves it before any deployment. Auto-suspend: a board that
        # replayed fast (crash recovery) forks nothing, so a dead run
        # races back through its cache serially, per her requirement.
        _pw = int(self.parameters.get('parallel_exec_workers', 0) or 0)
        _thr = float(self.parameters.get('warm_auto_threshold_s', 30)
                     or 0)
        # Board time = the interval between board STARTS - the whole
        # board, log dump and audit machinery included. The first
        # timer wrapped schedule.step only and undercounted 200s
        # boards as <20s, silently vetoing every fork (found by live
        # fire 2026-08-18: threshold 0.001 -> 8 children instantly).
        import time as _time3
        _nowb = _time3.monotonic()
        _prevb = getattr(self, '_board_started_at', None)
        self._board_started_at = _nowb
        _last = (_nowb - _prevb) if _prevb is not None else None
        if _pw > 1 and _last is not None and (_thr <= 0 or _last > _thr):
            # Brood sizing by what is ACTUALLY free (her memory law:
            # watch free RAM and the swap deficit, never "available";
            # and the gate-2 lesson: 12 COW children + neighbors =
            # kernel kills). Each child dirties ~1-1.5G on a fresh
            # board; keep a 6G reserve; under pressure this collapses
            # to a serial board instead of an OOM lottery.
            try:
                with open('/proc/meminfo') as _mf:
                    _mi = {}
                    for _ln in _mf:
                        if ':' in _ln:
                            _k, _v = _ln.split(':', 1)
                            _mi[_k] = int(_v.split()[0])
                _free_gb = _mi.get('MemFree', 0) / 1048576.0
                _swap_used_gb = (_mi.get('SwapTotal', 0)
                                 - _mi.get('SwapFree', 0)) / 1048576.0
                _avail_gb = _mi.get('MemAvailable', 0) / 1048576.0
            except (OSError, ValueError):
                _free_gb = _swap_used_gb = _avail_gb = 0.0
            # Two tiers (the first guard read MemFree and permanently
            # vetoed forks on any healthy cache box, whose page cache
            # rightly eats idle RAM): DISTRESS veto on true free +
            # swap deficit - her watch-the-thing law, catching real
            # thrash; CAPACITY sizing on MemAvailable, the kernel's
            # own reclaimable estimate, which is the correct question
            # for an allocation.
            if _free_gb < 2.0 or _swap_used_gb > 2.0:
                _pw = 0
            else:
                _pw = min(_pw, max(0, int((_avail_gb - 10.0) / 3.0)))
        _brood_lock = None
        if _pw > 1 and _last is not None and (_thr <= 0 or _last > _thr):
            # BOX BROOD LOCK (the 15:0x OOM wave): two co-resident
            # worlds brooding in the same seconds each saw the same
            # free memory and together blew the box. One brood per
            # machine: the loser of this non-blocking flock simply
            # runs its board serial and tries again next board.
            import fcntl as _fcntl2
            try:
                _brood_lock = open('.brood.lock', 'a')
                _fcntl2.flock(_brood_lock,
                              _fcntl2.LOCK_EX | _fcntl2.LOCK_NB)
            except OSError:
                try:
                    _brood_lock.close()
                except Exception:
                    pass
                _brood_lock = None
                _pw = 0
        if _pw > 1 and _last is not None and (_thr <= 0 or _last > _thr):
            import numpy as _np
            _rs = random.getstate()
            _ns = _np.random.get_state()
            _kids = []
            for _wi in range(_pw):
                try:
                    _pid = os.fork()
                except OSError:
                    break
                if _pid == 0:
                    try:
                        _null = os.open(os.devnull, os.O_WRONLY)
                        os.dup2(_null, 1)
                        os.dup2(_null, 2)
                        os.environ['MARKETSIM_WARM_CHILD'] = '1'
                        os.environ['MARKETSIM_WARM_SHARD'] = (
                            '%d/%d' % (_wi, _pw))
                        self.schedule.step()
                    except BaseException:
                        pass
                    finally:
                        os._exit(0)
                _kids.append(_pid)
            for _pid in _kids:
                try:
                    os.waitpid(_pid, 0)
                except OSError:
                    pass
            random.setstate(_rs)
            _np.random.set_state(_ns)
        if _brood_lock is not None:
            try:
                _brood_lock.close()
            except Exception:
                pass
        import time as _time2
        _t0s = _time2.monotonic()
        self.schedule.step()
        self._last_step_secs = _time2.monotonic() - _t0s

        # The audit institution (her corruption scenario, 2026-08-10;
        # the mathematics separated roles: anchored standards make the
        # world bistable, the audit is the TREATMENT that destabilizes
        # the corrupt attractor). Budget B audits per period spread
        # over the period's untested settles - the per-settle audit
        # probability is B over the pool size, so credulity DILUTES
        # enforcement (the documented congestion of real corruption).
        # A hit claws back a multiple of the OVERCHARGE (paid minus
        # the fair score-scaled price) - continuous, no thresholds.
        # No sign is stamped:
        # sign reputation must emerge by selection, not decree.
        _B = self.parameters.get('audit_budget', 0)
        _pool = getattr(self, '_audit_pool', [])
        if _B and _pool:
            _p_audit = min(1.0, float(_B) / len(_pool))
            _m = float(self.parameters.get('audit_penalty_multiplier', 2.0))
            for _buyer, _buy, _wn, _paid, _seller_uid, _frontier in _pool:
                if random.random() >= _p_audit:
                    continue
                try:
                    _base = stop_cut(str(_buy.get('item', '')))
                    _auth = []
                    for _mt in getattr(_buyer, 'initial_trade_plan',
                                       {}).get('trades', []):
                        if stop_cut(str(_mt.get('item', ''))) == _base:
                            _auth = [dict(_t) for _t in _mt.get('tests', [])
                                     if _t.get('test')
                                     and not _t.get('stophere')]
                            break
                    if not _auth:
                        continue
                    _shadow = dict(_buy)
                    _shadow['tests'] = _auth + [{'stophere': True}]
                    _sp, _sg, _truth, _sn = _buyer.pass_all_tests(_shadow, _wn)
                    if _frontier > 0 and _truth < _frontier:
                        _claw = _m * _paid * (1.0 - _truth / _frontier)
                        _seller = None
                        for _ag in self.schedule.agents:
                            if _ag is not None and \
                                    _ag.unique_id == _seller_uid:
                                _seller = _ag
                                break
                        if _seller is not None:
                            _seller.agiTokens -= _claw
                            print('AUDIT HIT iter %s: seller %s truth %.4f '
                                  'frontier %.4f paid %.2f clawback %.2f'
                                  % (getattr(self.schedule, 'time', -1),
                                     _seller_uid, _truth, _frontier,
                                     _paid, _claw))
                except Exception as _ae:
                    print('AUDIT ERROR: %s' % _ae)
        self._audit_pool = []

        # HER RULINGS (2026-08-13, "first yes, second yes" and then
        # "what good would care do"): release the per-step bookkeeping.
        #
        # Both of these dictionaries were growing for the life of the run
        # and neither has a reader in a later step.
        #
        # self.pickles maps a chain key to its cache filename. It is not
        # used to find anything - a cache hit is decided by whether the
        # file exists on disk - and its only consumer is
        # buy['pickle'] = model.pickles.get(key), which runs in the same
        # step that stored it. Once the board carries the filename as a
        # plain string the mapping is spent.
        #
        # emergent_functions (and its arity twin) holds an assembled chain
        # under a freshly minted f<N>_ name. Every reader - the arity
        # lookup, modular_gep, the line that consumes the name right after
        # creating it, and item_type through convert_to_cumulative_category
        # - runs inside the audition that created it. The boards carry
        # ontology names and fully expanded code, never these ids: no sell
        # listing is f-prefixed. I had assumed entries needed protecting
        # and proposed reference counting; her question ("what good would
        # care do") was right - there is no lifetime to protect.
        #
        # Measured before this: the inherited index alone was 1,246,309
        # entries and 0.88 GB resident, and a run reached 5.7 GB against
        # her flat 2-3 GB invariant.
        self.pickles.clear()
        self.emergent_functions.clear()
        self.emergent_functions_arity.clear()

        # MEMORY DIAGNOSTIC (her order, 2026-08-13: "debug why we are blowing
        # up ram"). Off unless MARKETSIM_MEMDIAG is set, and deliberately
        # expensive - it walks every tracked object - so it is a diagnostic
        # run's tool, never a science run's.
        #
        # The question it answers in one column pair: is resident memory
        # RETAINED or merely the high-water mark of transients? live_array_MB
        # is the total held by numpy arrays that are still reachable. If it
        # stays flat while rss_MB climbs, nothing is retained and the growth
        # is allocation behaviour. If both climb together, something is
        # holding arrays and the other columns say which container.
        if os.environ.get('MARKETSIM_MEMDIAG'):
            try:
                import gc as _gc
                import numpy as _np
                with open('/proc/self/statm') as _sf:
                    _rss = int(_sf.read().split()[1]) * 4096
                _objs = _gc.get_objects()
                _arrs = [o for o in _objs if isinstance(o, _np.ndarray)]
                _nb = 0
                for _a in _arrs:
                    try:
                        if _a.base is None:      # count owners, not views
                            _nb += _a.nbytes
                    except Exception:
                        pass
                _p = self.parameters['output_path'] + 'memdiag.csv'
                _new = not os.path.exists(_p)
                with open(_p, 'a') as _h:
                    if _new:
                        _h.write('time;rss_MB;live_objects;live_arrays;'
                                 'live_array_MB;cache;unpickle_memo;pickles;'
                                 'emergent\n')
                    _h.write('%s;%.1f;%d;%d;%.1f;%d;%d;%d;%d\n' % (
                        self.schedule.time, _rss / 2**20, len(_objs),
                        len(_arrs), _nb / 2**20, len(self.cache),
                        len(getattr(self, '_unpickle_memo', ()) or ()),
                        len(self.pickles), len(self.emergent_functions)))
            except Exception as _me:
                print('memdiag failed: %s' % _me)

    def _disk_free_gb(self, path=None):
        """Free gibibytes on the filesystem the run writes to."""
        try:
            st = os.statvfs(path or self.parameters.get('output_path') or '.')
            return (st.f_bavail * st.f_frsize) / float(1 << 30)
        except OSError:
            return float('inf')

    def _disk_preflight(self):
        """HER RULING c9 (2026-08-13): the engine had no disk check of any
        kind - it capped memory, evaluation time and solver iterations, and
        nothing at all watched the filesystem. The shared cache therefore
        grew unbounded to 305 GiB, and because WSL's virtual disk expands on
        the Windows volume as WSL fills, an unbounded cache took C: down to
        14 GB from the inside.

        A refusal is only safe BEFORE anything is running, so it lives here.
        The two numbers are mine and here is why they are what they are.
        FLOOR (80 GiB): a wave-scale run's cache growth between the checks a
        human actually makes; below it, writing more cache is what turns a
        tight disk into a dead machine. STOP (25 GiB): the level at which the
        host itself is at risk rather than the run - C: was at 14 GB when
        this became a crisis. Both are config keys; neither is hidden."""
        floor = float(self.parameters.get('disk_floor_gb', 80))
        free = self._disk_free_gb()
        if free < floor:
            raise SystemExit(
                'REFUSING TO LAUNCH: %.1f GiB free, floor is %.1f GiB.\n'
                'Nothing is running yet, so nothing is lost by stopping here.\n'
                'Free space or lower disk_floor_gb in the config deliberately.'
                % (free, floor))
        print('disk preflight OK: %.1f GiB free (floor %.1f)' % (free, floor))
        # LAUNCH MUTEX (her order 2026-08-18: "you have an exact
        # replicate up and it blows memory. fix that"). A second
        # instance of the SAME world doubled memory until the kernel
        # killed something, repeatedly. flock on the experiment dir:
        # advisory, crash-safe (the lock dies with the process, so no
        # stale locks), held for the process lifetime via self.
        import fcntl as _fcntl
        _lockdir = self.parameters['output_path']
        os.makedirs(_lockdir, exist_ok=True)
        _lockpath = os.path.join(_lockdir, '.run.lock')
        self._run_lock_fd = open(_lockpath, 'a+')
        try:
            _fcntl.flock(self._run_lock_fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(
                'REFUSING TO LAUNCH: %s is already running on this '
                'machine (lock %s is held). Nothing is running yet '
                'from THIS process, so nothing is lost by stopping '
                'here.' % (self.parameters.get('label', '?'), _lockpath))
        self._run_lock_fd.seek(0)
        self._run_lock_fd.truncate()
        self._run_lock_fd.write('%d\n' % os.getpid())
        self._run_lock_fd.flush()

    def _disk_check(self, i):
        """Called every iteration. Below the floor it suspends cache WRITES,
        which costs speed and changes no result: the evaluation sandbox is
        deterministic from (chain, scenario seed), so a chain recomputed
        later returns exactly what the cache would have returned. Below the
        stop level it ends the run cleanly - her ruling c6, 2026-08-13, 'you
        may stop a run for this reason', the single exception to the standing
        rule that a run is never stopped."""
        # An outside watcher may ask for a clean stop by dropping this file.
        # It is honoured HERE, between iterations, so a run always ends on a
        # written board rather than mid-write. The watcher needs this because
        # it can see the Windows volume, which this process cannot: WSL's
        # virtual disk grows on C: as WSL fills, so the host can be dying
        # while the run's own filesystem still looks comfortable.
        _sf = self.parameters.get('stop_file') or 'STOP_THIS_RUN'
        # Two homes are honoured: the launch directory (one file there
        # stops every run launched from it) and the run's own output
        # directory (vitals.sh drops the file into the experiment dir on
        # a disk alarm - the c6 exception; until 2026-08-14 the engine
        # only looked in the launch directory, so that lever silently
        # missed).
        for _sfp in (_sf,
                     os.path.join(self.parameters['output_path'], _sf)):
            try:
                if os.path.exists(_sfp):
                    with open(_sfp) as _h:
                        _why = _h.read().strip()[:200]
                    print('CLEAN STOP requested at iteration %d: %s'
                          % (i, _why or 'no reason given'))
                    self._disk_stop_reason = _why or 'stop file'
                    return False
            except OSError:
                pass

        floor = float(self.parameters.get('disk_floor_gb', 80))
        stop = float(self.parameters.get('disk_stop_gb', 25))
        free = self._disk_free_gb()
        if free < stop:
            print('DISK STOP at iteration %d: %.1f GiB free, below %.1f. '
                  'Ending the run cleanly.' % (i, free, stop))
            self._disk_stop_reason = ('%.1f GiB free at iteration %d'
                                      % (free, i))
            return False
        suspend = free < floor
        if suspend and not getattr(self, '_cache_writes_suspended', False):
            print('DISK ALARM at iteration %d: %.1f GiB free, below the %.1f '
                  'floor. Cache writes suspended; the run continues.'
                  % (i, free, floor))
        elif (not suspend) and getattr(self, '_cache_writes_suspended', False):
            print('disk recovered at iteration %d: %.1f GiB free. Cache '
                  'writes resumed.' % (i, free))
        self._cache_writes_suspended = suspend
        return True

    def go(self):
        self._disk_preflight()
        for i in range(self.parameters['max_iterations']):
            if not self._disk_check(i):
                break
            print("iteration " + str(i))
            self._iter = i
            if self.parameters.get('judged_per_iteration'):
                # One judgment per iteration, allocated by SIGN MATCH
                # (the curator, 2026-08-06): one society-wide weighted draw
                # over buyer-want and seller pairs, chance proportional
                # to sign closeness minus the field minimum, the same
                # law as the within-want chooser, so better signs earn
                # more attention while every displayed seller stays
                # reachable. 'price' and 'uniform' remain as config
                # alternatives over wants alone.
                _mode = str(self.parameters.get('judgment_lottery',
                                                'sign')).lower()
                _pairs = []
                for _ai, _m in enumerate(self.blackboard):
                    for _t in _m.get('trades', []):
                        if _t.get('type') != 'buy':
                            continue
                        _act = False
                        for _td in _t.get('tests', []):
                            if _td.get('stophere'):
                                break
                            if not (stop_cut((_td.get('test') or ''))
                                    and stop_cut((_td.get('data') or ''))):
                                break
                            _act = True
                            break
                        if not _act:
                            continue
                        _pref = stop_cut(str(_t.get('item', '')))
                        if _mode != 'sign':
                            _w = (float(_t.get('midpoint', 1.0) or 1.0)
                                  if _mode == 'price' else 1.0)
                            _pairs.append(((_ai, _pref, None), _w))
                            continue
                        _sought = _t.get('sign')
                        if not isinstance(_sought, list) or not _sought:
                            continue
                        for _si, _sm in enumerate(self.blackboard):
                            if _si == _ai:
                                continue
                            for _st in _sm.get('trades', []):
                                if _st.get('type') != 'sell':
                                    continue
                                if not str(_st.get('item', '')).startswith(_pref):
                                    continue
                                if self.parameters.get('judgment_overlap_filter'):
                                    # Optional band-overlap filter. OFF
                                    # by her ruling (2026-08-06): the
                                    # lottery is pure sign match, and a
                                    # judgment of an unsettleable pair
                                    # is allowed to burn the slot.
                                    _bm = float(_t.get('midpoint', 0) or 0)
                                    _br = float(_t.get('range', 0) or 0)
                                    _sm2 = float(_st.get('midpoint', 0) or 0)
                                    _sr = float(_st.get('range', 0) or 0)
                                    if _bm + _br < max(_sm2 - _sr, 0.0) \
                                            or _sm2 + _sr < max(_bm - _br, 0.0):
                                        continue
                                _disp = _st.get('sign')
                                if not isinstance(_disp, list) or not _disp:
                                    continue
                                _n = min(len(_sought), len(_disp))
                                _dot = _na = _nb = 0.0
                                for _k in range(_n):
                                    _x = float(_sought[_k]); _y = float(_disp[_k])
                                    _dot += _x * _y; _na += _x * _x; _nb += _y * _y
                                if _na > 0 and _nb > 0:
                                    _pairs.append(
                                        ((_ai, _pref, _si),
                                         _dot / ((_na ** 0.5) * (_nb ** 0.5))))
                self._judged_key = None
                self._judged_seller = None
                if _pairs:
                    if _mode == 'sign':
                        _mn = min(w for _, w in _pairs)
                        _pairs = [(k, w - _mn) for k, w in _pairs]
                        if sum(w for _, w in _pairs) <= 0:
                            _pairs = [(k, 1.0) for k, _ in _pairs]
                    _r = random.random() * sum(w for _, w in _pairs)
                    _pick = _pairs[-1][0]
                    for _k, _w in _pairs:
                        _r -= _w
                        if _r <= 0:
                            _pick = _k
                            break
                    self._judged_key = (_pick[0], _pick[1])
                    self._judged_seller = _pick[2]
            self.step()
            if self.parameters.get('sign_swap_at') is not None \
                    and int(self.parameters.get('sign_swap_at')) == i:
                # THE SIGN-SWAP INTERVENTION (3 Oct 2026, flag-gated): at this board
                # the learners' DISPLAY signs are swapped pairwise in their search
                # state (the CMA-ES mean and the pending candidates) while every
                # role, sought sign and price stays. If the division of labour runs
                # through the signs, service collapses on the next board and
                # recovers as buyers re-point: the latch time, measured directly.
                import numpy as _np
                _ss = int(self.parameters.get('sign_size', 16))
                _ls = sorted((_a for _a in self.schedule.agents
                              if _a.__class__.__name__ == 'RoleSim'
                              and not getattr(_a, 'initial_trade_plan', None)),
                             key=lambda _a: _a.unique_id)
                _n_sw = 0
                for _k in range(0, len(_ls) - 1, 2):
                    _a, _b = _ls[_k], _ls[_k + 1]
                    try:
                        _ma, _mb = _np.array(_a.es.mean, dtype=float), _np.array(_b.es.mean, dtype=float)
                        _a.es.mean = _np.concatenate([_mb[:_ss], _ma[_ss:]])
                        _b.es.mean = _np.concatenate([_ma[:_ss], _mb[_ss:]])
                        for _c in range(min(len(_a.solutions), len(_b.solutions))):
                            _sa = _np.array(_a.solutions[_c], dtype=float); _sb = _np.array(_b.solutions[_c], dtype=float)
                            _a.solutions[_c] = _np.concatenate([_sb[:_ss], _sa[_ss:]])
                            _b.solutions[_c] = _np.concatenate([_sa[:_ss], _sb[_ss:]])
                        _n_sw += 1
                    except Exception as _se:
                        print('sign swap failed for %d/%d: %s' % (_a.unique_id, _b.unique_id, _se))
                print('SIGN SWAP at iteration %d: %d pairs' % (i, _n_sw))
            if self.parameters.get('birth_death'):
                # Birth and death as separate RATES (the curator, 2026-08-06):
                # no wealth conditioning, no replacement coupling, all
                # else identical to the plain society. Each unmasked
                # learner dies with probability death_rate per iteration,
                # chosen by chance alone; newborns arrive at birth_rate
                # expected per iteration with fresh genomes from the
                # broad priors, ADDED to the population, so its size
                # floats. Authored humans never die. A death tombstones
                # the agent's board slot so indexing stays aligned; a
                # birth appends a new slot. Both count as novelty.
                _dr = float(self.parameters.get('death_rate', 0.002))
                _br = float(self.parameters.get('birth_rate', 0.4))
                for _ag in list(self.schedule.agents):
                    if _ag.__class__.__name__ not in ('RoleSim', 'LLMAgent'):   # language-model learners die too (7 Oct)
                        continue
                    if getattr(_ag, 'initial_trade_plan', None):
                        continue
                    if random.random() < _dr:
                        _uid = _ag.unique_id
                        self.schedule.remove(_ag)
                        self.blackboard[_uid] = OrderedDict([
                            ('type', 'RoleSim'),
                            ('label', 'DEAD Agent %d' % _uid),
                            ('trades', [])])
                        self._last_novelty_iter = i
                        print('DEATH agent %d at iteration %d' % (_uid, i))
                _births = int(_br)
                if random.random() < (_br - _births):
                    _births += 1
                _cap = self.parameters.get('max_learners')
                if _cap:
                    # Memory-doctrine cap: births pause at the cap and
                    # deaths reopen room, so population floats under a
                    # pinned ceiling instead of growing to the rate
                    # equilibrium.
                    _live = sum(
                        1 for _a2 in self.schedule.agents
                        if _a2.__class__.__name__ in ('RoleSim', 'LLMAgent')
                        and not getattr(_a2, 'initial_trade_plan', None))
                    _births = max(0, min(_births, int(_cap) - _live))
                for _ in range(_births):
                    _uid = len(self.blackboard)
                    # a newborn is of the population's own kind (7 Oct 2026): language-model learners beget language-model learners
                    _kind = 'LLMAgent' if 'LLMAgent' in (self.parameters.get('random_agents') or {}) else 'RoleSim'
                    _apar = self.parameters['agent_parameters'].get(_kind)
                    _nb = (LLMAgent if _kind == 'LLMAgent' else RoleSim)(_uid, self, None, _apar)
                    self.schedule.add(_nb)
                    self._last_novelty_iter = i
                    print('BIRTH agent %d at iteration %d' % (_uid, i))
            # Convergence stop with room (the curator, 2026-08-05): never
            # stop at first flatness. A plateau counts only when no
            # novelty signal fires, and novelty is any frontier record,
            # any never-before-traded item (both marked at settlement),
            # or a settlement rate that moved more than ten percent
            # between checks. The stop needs the plateau to persist a
            # long confirmation span, never before a minimum age, and
            # any single novelty resets the countdown. All thresholds
            # are config keys so the room can widen without code.
            _ck = self.parameters.get('convergence_check_every', 250)
            # Debug worlds may disable the stop entirely (her option,
            # 2026-08-12): config-only, default ENABLED - the
            # convergence law for science worlds is untouched; a
            # debug world runs until stopped by hand.
            if self.parameters.get('convergence_stop_enabled', True) \
                    and _ck and i >= self.parameters.get('convergence_min_age', 4000) \
                    and i % _ck == 0:
                _count = getattr(self, '_settle_count', 0)
                _prev = getattr(self, '_conv_prev_window', None)
                _delta = _count - getattr(self, '_conv_prev_count', 0)
                self._conv_prev_count = _count
                self._conv_prev_window = _delta
                if _prev is not None and abs(_delta - _prev) > 0.1 * max(_prev, 1):
                    self._last_novelty_iter = i
                _quiet = i - getattr(self, '_last_novelty_iter', 0)
                if _quiet >= self.parameters.get('convergence_confirm_iters', 2000):
                    print('CONVERGED_STOP at iteration %d '
                          '(quiet for %d iterations)' % (i, _quiet))
                    break


# (the thread constant that used to live here is gone - her 2026-08-13
#  correction made thread pinning an option, resolved from the config)



def _pin_hash_seed(config_path=None):
    """HER RULING (2026-08-13): "Make pinning python seed part of the hard
    coding of the run. It should not be an option that the run not be
    reproducible."

    Python randomises the hash of strings per process unless PYTHONHASHSEED
    is fixed. The engine iterates SETS of function names - the emergent
    subroutines gathered in modular_gep - and a set of strings is walked in
    hash order, so two launches of an otherwise identical configuration can
    walk them differently. Every other source of variation is already
    pinned: each agent's search stream is a pure function of (config seed,
    agent identity, build ordinal), every global generator is seeded from
    the config, and stochastic products draw from a sandbox keyed by the
    chain rather than the global stream. This was the one variable the
    design could not reach from inside itself.

    It cannot be fixed by setting os.environ here: the interpreter has
    already chosen its hash seed by the time any of our code runs. The only
    way to make it non-optional is to re-exec with the variable set, which
    is what this does - once, before any work, discarding the process that
    had the wrong seed. The cost is paying module import twice on launch.

    The value is the CONFIG SEED, not a constant - her correction, and the
    right design. A constant pins every run to the same walk order, which
    correlates runs that are meant to be independent. Taking it from the
    seed gives one hash seed per run seed: the same run always reproduces,
    and two runs with different seeds differ everywhere they should.

    HER RULING, same day, on the thread count: "hard code it into the code
    without a config option not to pin it." So the thread count is pinned
    here too, and there is deliberately NO parameter that can switch either
    pin off. Thread count belongs in this function for a reason beyond the
    ruling: the numeric libraries fix their thread pools when they are
    imported, which happens before main() runs, so it could not be pinned
    from inside a started process at all. The re-exec sets both before any
    import in the replacement process.

    THE THREAD VALUE IS MINE, not hers, and it is a FIXED CONSTANT rather
    than a fraction of the machine's cores. A fraction would make results
    depend on which machine ran them, and the number of threads changes the
    order floating-point sums accumulate, so a fraction would buy
    reproducibility on one box and lose it between boxes. Two is low enough
    that several runs coexist on a laptop and high enough not to cripple a
    single one. Change the constant if the hardware changes; there is no way
    to unset it.
    """
    # HER CORRECTION (2026-08-13): "you are not supposed to pin the python
    # hash seed with a constant number for all runs, you are supposed to pin
    # it with a number that comes from the runs seed, so that it is the same
    # pin for the same run seed but different pins for different seeds."
    #
    # A constant would make every run walk its sets of function names in the
    # SAME order, which correlates runs that are supposed to be independent
    # draws. Deriving it from the config seed gives both properties at once:
    # one seed always produces one hash seed, so a run reproduces; two seeds
    # produce two, so two runs differ everywhere they should.
    #
    # The config has to be read here, before the re-exec, because after it
    # the interpreter's hash seed is already fixed. It is a small JSON read
    # and the file is opened again by __init__ moments later. If it cannot
    # be read the run does not start: a run that cannot be pinned is a run
    # that is not reproducible, and that is not an available outcome.
    _cfg = config_path or (sys.argv[1] if len(sys.argv) > 1 else 'study.json')
    try:
        with open(_cfg) as _f:
            _params = json.load(_f)['parameters']
        _seed = int(_params['seed'])
    except (OSError, KeyError, ValueError, TypeError) as _e:
        raise SystemExit('cannot pin the hash seed: no readable seed in %s '
                         '(%s). A run that cannot be pinned is not '
                         'reproducible, so it does not start.' % (_cfg, _e))
    pins = {'PYTHONHASHSEED': str(_seed % 4294967296)}

    # HER CORRECTION (2026-08-13) on the threads, overriding my hard-coding:
    # "i dont believe you on the threads, i think you are making it so that we
    # can not fill up available cores. so i want to change it so that pinning
    # the threads is an option, so when the option is off it has the current
    # behavior of filling up available cores, and then how many cores to pin
    # it on is an option. many people doent like to know their machine at that
    # level and can not set that number effectively."
    #
    # She is right about the cost: pinned at two, one run used two of fourteen
    # cores and the rest of the machine sat idle. Extending her hash-seed
    # ruling to threads was mine and it was wrong.
    #
    # DEFAULT OFF: the numeric libraries fill the machine exactly as before.
    # pin_threads on makes the run bit-comparable across launches and boxes;
    # threads_per_run says how many, and when it is omitted the number comes
    # from the machine so nobody has to know their hardware to use the switch.
    # Half the cores is mine: it leaves room for a second run without either
    # one being starved.
    if _params.get('pin_threads'):
        _n = _params.get('threads_per_run')
        if not _n:
            _n = max(1, (os.cpu_count() or 4) // 2)
        _n = str(int(_n))
        pins.update({'OMP_NUM_THREADS': _n, 'MKL_NUM_THREADS': _n,
                     'OPENBLAS_NUM_THREADS': _n, 'NUMEXPR_NUM_THREADS': _n})

    if all(os.environ.get(k) == v for k, v in pins.items()):
        return
    os.environ.update(pins)
    print('pinning %s and re-executing so the run is reproducible '
          '(her rulings 2026-08-13)'
          % ', '.join('%s=%s' % kv for kv in sorted(pins.items())))
    sys.stdout.flush()
    os.execv(sys.executable, [sys.executable] + sys.argv)


def main():
    _pin_hash_seed()
    marketsim = MarketSim(sys.argv[1]) if len(sys.argv) > 1 else MarketSim()
    marketsim.go()


if __name__ == '__main__':
    main()
