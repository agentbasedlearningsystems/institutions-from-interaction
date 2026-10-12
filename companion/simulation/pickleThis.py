"""pickleThis — content-addressed cache for chain results.

Filename = sha256(str(chain_key))[:32] + '.p'. Same chain key always
hashes to the same filename, so multiple processes can safely share a
pickles/ directory: they each write the same bytes to the same path,
and atomic-rename via os.replace prevents partial-read races.

Sharing model (introduced May 21 2026):
  - Each scenario can either:
    (a) keep its own per-scenario ``pickles/`` directory (no sharing)
    (b) symlink its ``pickles/`` directory to a shared location, e.g.
        ``experiments/cc1_shared_pickles/``, so multiple scenarios all
        write to and read from the same content-addressed namespace
  - Either choice is safe because the filename is deterministic from
    the chain key — no counter race possible.

The legacy ``self.pickles`` dict and ``self.pickle_count`` are still
present on the MarketSim instance (for backward compatibility with the
MarketSim init code that loads ``index.p``) but they are no longer
consulted by ``pickleThis``. Old numbered pickles in pre-existing
scenario dirs become unreadable by the new code; migrating them is a
separate one-shot script (``scripts/migrate_pickles_to_shared.py``).

Processes started before this change keep using the old code in their
own memory and continue writing integer-named files to their own dir
— they don't see the new content-addressed namespace, but they also
don't conflict with it.
"""
import hashlib
import os
import pickle
from collections import OrderedDict


# Env vars whose value affects the simulation's output and therefore
# must be part of the cache key. Setting one of these to a non-default
# value (e.g. MARKETSIM_VECTORSPACE_MAX_ROWS=10000) needs to produce
# DIFFERENT cache files than the unset-default behavior, otherwise we
# silently mix incompatible cached results.
#
# Backward compatibility note: if NONE of these env vars are set, we
# fall back to the original hash payload (just str(key)) so that
# pickles written by the pre-2026-05-21 code remain reachable. Pickles
# written with one of these vars explicitly set get a different hash
# even if the value matches the historical default.
_CACHE_NAMESPACE_ENV_VARS = (
    "MARKETSIM_VECTORSPACE_MAX_ROWS",   # rows passed to clusterers
    "CORE_DATA_NROWS",                # rows loaded from CSV corpora
)


def _key_to_filename(key):
    """Deterministic content-addressed filename for a chain key.
    Uses sha256 truncated to 32 hex chars + '.p' suffix → 34-char names,
    collision-free for any realistic chain-key space.

    If any cache-namespace env var is set, its name=value is folded into
    the hash payload so that runs with different env values produce
    distinct cache entries. If none are set, payload is just str(key) —
    matches the pre-namespacing hash, so historical pickles remain
    reachable.
    """
    # HER RULING, 2026-08-14: "make the shared cache method the sha of the
    # gep string, like you say it is."
    #
    # It is the sha of the GEP string and NOTHING ELSE. The GEP string names
    # every node the program uses - the test, the algorithm with its
    # parameters, the labeler, the corpus - so it identifies the whole
    # program; and MarketSim's evaluation sandbox derives its randomness from
    # that same string (`seed_material = f"{tuple_key}"`), so one program
    # gives one output on every machine, in every seed's world, forever.
    # A chain therefore has exactly ONE address, and anything appended to
    # the name can only invent a second address for one thing.
    #
    # Two things used to be appended and both are gone:
    #   - environment values (row caps), which made the same program land
    #     in different files depending on how the shell was set up;
    #   - '|era=postprefab' on any chain containing a labeler, which split
    #     the labeler keyspace in two. Her ruling the same day - "no
    #     prefabs please" - removes the thing that tag was guarding.
    # Names written under the old recipes are simply stale: nothing
    # computes them any more, and their chains recompute to the one
    # correct address.
    return hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:32] + ".p"


class _EvalTimeout(Exception):
    pass


def _eval_alarm(signum, frame):
    raise _EvalTimeout()


def pickleThis(fn):  # decorator
    def wrapped(self, *args, **kwargs):
        # No key, nothing to cache — just call through.
        if not args:
            return fn(self, *args, **kwargs)
        key = args[0]
        # THE NAME IS THE CHAIN, AND ONLY THE CHAIN (her decision a1,
        # 2026-08-12). I had briefly folded the scenario seed into the
        # name - ('SEMV2', seed, key) - reasoning that a seed's world
        # must not be served another seed's draws. That broke her
        # design: "there is a cache for all runs, that accumulates with
        # every run and is used by all runs". Measured cost of the
        # mistake: 95% of the 1,061,504 recorded chains were reachable
        # only under the chain-keyed name, so the seed-tagged world
        # recomputed a 232GB store it already owned - which is why the
        # run crawled and why the cache looked unwritten.
        #
        # First-writer-canonical is what makes one name correct: a
        # chain's value is whatever the first evaluator recorded, for
        # every society, every seed and every machine, forever. The
        # scenario seed therefore has no business in the name; it can
        # only decide the value of a chain nobody has computed yet, and
        # decision a2 (deriving a chain's randomness from the chain
        # alone) is what will make even that principled.
        pickle_name = _key_to_filename(key)
        # Single shared cache across societies (the curator, 2026-08-07):
        # cache_path, when set, points every society at one store;
        # content-addressed names make cross-society hits exact.
        pickle_dir = self.parameters.get('cache_path') \
            or (self.parameters['output_path'] + 'pickles/')
        cachefile = pickle_dir + pickle_name

        # Cache hit: read and return
        if os.path.exists(cachefile):
            try:
                # In-memory memo over the disk cache (profile: 21k
                # pickle.load calls, 154 cumulative seconds in a
                # two-iteration probe). Bounded; oldest evicted.
                _memo = getattr(self, '_unpickle_memo', None)
                if _memo is None:
                    _memo = self._unpickle_memo = OrderedDict()
                if pickle_name in _memo:
                    _memo.move_to_end(pickle_name)
                    res = _memo[pickle_name]
                    if hasattr(self, 'pickles'):
                        self.pickles[key] = pickle_name
                    return res
                with open(cachefile, 'rb') as h:
                    res = pickle.load(h)
                _memo[pickle_name] = res
                # BYTE CAP (2026-09-12): the count cap alone let this memo
                # hold 128 multi-hundred-MB deliveries - RSS 10-14 GB per
                # world, the machine exhausted twice with two worlds up.
                # Entry size = the pickle's size on disk (a fair proxy);
                # oldest evicted until under unpickle_memo_bytes (config,
                # default 1.5e9). A pure cache over the disk store: no
                # result changes, only how many reloads happen.
                _sz = getattr(self, '_unpickle_memo_sz', None)
                if _sz is None:
                    _sz = self._unpickle_memo_sz = {}
                try:
                    _sz[pickle_name] = os.path.getsize(cachefile)
                except OSError:
                    _sz[pickle_name] = 0
                _cap = float(self.parameters.get('unpickle_memo_bytes', 1.5e9))
                while _memo and (len(_memo) > 128
                                 or sum(_sz.get(k, 0) for k in _memo) > _cap):
                    _k, _ = _memo.popitem(last=False)
                    _sz.pop(_k, None)
                # Cache-value law (her order, 2026-08-18): record hits
                # so the nightly pruner can score seconds x hits per
                # byte. Buffered; a dropped tail at exit only rounds
                # a count down.
                _hb = getattr(self, '_cache_hit_buf', None)
                if _hb is None:
                    _hb = self._cache_hit_buf = []
                if not os.environ.get('MARKETSIM_WARM_CHILD'):
                    # warm children re-hit entries W times over; their
                    # counts would distort the pruner's value law
                    _hb.append(pickle_name)
                if len(_hb) >= 200:
                    try:
                        with open(pickle_dir.rstrip('/').rsplit('/', 1)[0]
                                  + '/cache_hits.log', 'a') as _hf:
                            _hf.write('\n'.join(_hb) + '\n')
                    except OSError:
                        pass
                    del _hb[:]
                # Keep the legacy pickle index populated on HITS too:
                # buy['pickle'] (and thus recorded bought_items chains in
                # the reproduction report) depends on this mapping, and
                # warm-cache societies otherwise record no chains at all
                # (Society A: 269 settles, 3 recorded chains).
                if hasattr(self, 'pickles'):
                    self.pickles[key] = pickle_name
                return res
            except (EOFError, pickle.UnpicklingError, MemoryError):
                # Partial write from a crashed process — or a cached result
                # too large to load under MARKETSIM_EVAL_MEMCAP_GB — recompute
                # below (the compute path's MemoryError is caught by
                # memoise_pickle and scored None).
                pass

        # Cache miss: compute, then write atomically (tmp + rename).
        _shard = os.environ.get('MARKETSIM_WARM_SHARD')
        if _shard:
            # Sharded speculative warm (her cores order, 2026-08-18):
            # each warm child computes only its hash-slice of the
            # board's fresh chains and raises on the rest - drawn sets
            # are fixed by signs and the cloned RNG, so later requests
            # do not depend on the skipped results. PYTHONHASHSEED is
            # pinned (306), so slices agree across children.
            _si, _sw = _shard.split('/')
            if hash(pickle_name) % int(_sw) != int(_si):
                raise RuntimeError('warm-shard-skip')
        import time as _time
        _t0 = _time.monotonic()
        _cap = None
        try:
            _cap = float(getattr(self, 'parameters', {}).get('evaluation_timeout_seconds', 0) or 0)
        except Exception:
            _cap = 0
        # a22, HER RULING (2026-08-12): "caches can be per box or even
        # per class of box, but not per run within the box. define them
        # at first. this one is labeled for getac class."
        #
        # A timeout is the one outcome that is NOT a fact about the
        # chain - it is a fact about how fast the machine is. Recording
        # it in the chain-keyed store would let a slow box mark a good
        # chain dead for every society and machine forever, which is
        # exactly what a1 and a2 removed this morning. Recording it
        # PER RUN would be worse: the same world would behave
        # differently on its second pass.
        #
        # So a timeout is recorded against the BOX CLASS, named up front
        # in the config as box_class (this machine: getac). Another
        # class ignores it and may compute the value; if it succeeds,
        # the value lands in the shared chain-keyed store and everyone
        # gains. Within a class it is canonical, so the cap is paid once
        # rather than every time the chain is met.
        _boxcls = str(getattr(self, 'parameters', {}).get('box_class') or '').strip()
        _tofile = None
        if _boxcls:
            _tofile = pickle_dir + _key_to_filename(('TIMEOUT', _boxcls, key))
            if os.path.exists(_tofile):
                # This class already learned it cannot finish this chain.
                if hasattr(self, 'pickles'):
                    self.pickles[key] = pickle_name
                return None
        if _cap and _cap > 0:
            import signal as _sig
            _old = _sig.signal(_sig.SIGALRM, _eval_alarm)
            _sig.alarm(int(_cap))
            _timed_out = False
            try:
                res = fn(self, *args, **kwargs)
            except _EvalTimeout:
                # Evaluation time cap (the curator, 2026-08-07): a product
                # too expensive to judge fails its audition like a
                # broken chain, mirroring the memory cap.
                res = None
                _timed_out = True
            finally:
                _sig.alarm(0)
                _sig.signal(_sig.SIGALRM, _old)
            if _timed_out:
                if _tofile:
                    try:
                        _tmp = _tofile + '.tmp.%d' % os.getpid()
                        with open(_tmp, 'wb') as _th:
                            pickle.dump({'chain': str(key), 'box_class': _boxcls,
                                         'cap_seconds': _cap}, _th)
                        os.replace(_tmp, _tofile)
                    except OSError:
                        pass
                # NEVER into the chain-keyed store: this says nothing
                # about the chain.
                if hasattr(self, 'pickles'):
                    self.pickles[key] = pickle_name
                return None
        else:
            res = fn(self, *args, **kwargs)

        _skipz = getattr(self, 'parameters', {}).get('cache_skip_cheap_zeros')
        if _skipz:
            # Cache-flood guard (2026-08-08 late): under sign-only
            # matching most auditions are wrong-audience TEST results
            # scoring zero in milliseconds; persisting each one grew
            # the store ~40k files/hour and the host disk with it.
            # Real scores are always cached; a zero/None TEST result
            # is cached only if it was expensive to compute (worth
            # not repeating). Component outputs are never filtered.
            _elapsed = _time.monotonic() - _t0
            _thresh = 1.0 if _skipz is True else float(_skipz)
            _zero = res is None
            if (not _zero) and isinstance(res, tuple) and len(res) == 2 \
                    and isinstance(res[1], (int, float)):
                _zero = res[1] <= 0
            if _zero and 'test_' in key and _elapsed < _thresh:
                return res

        # HER RULE, 2026-08-14: "things that come back immediately with a
        # none and take about the same time as looking in the cache dont
        # need to be in the cache. its got to save computation time to be
        # in the cache."
        #
        # A cache entry only earns its place if reading it back is cheaper
        # than working the answer out again. MEASURED on this store, 300
        # random files: a hit costs 0.768 ms at the median, 1.07 ms at the
        # ninetieth percentile - open, read, unpickle. So a computation
        # that finishes in under a millisecond is not worth storing: the
        # lookup would cost as much as the work, and the entry then makes
        # every FUTURE lookup slower, because it enlarges the one flat
        # directory that every probe walks. That directory is presently
        # 151 MB of entries across 2.33 million files, and 93% of those
        # files hold None, 0.0 or an empty array - results that cost
        # nothing to produce and are stored forever.
        #
        # This subsumes cache_skip_cheap_zeros, which asked a narrower
        # question: it skipped only zero-ish results, only under 'test_'
        # keys, and left a fast None under any other key stored at full
        # price. The rule below does not care what the value is - only
        # whether keeping it saves work.
        _floor = getattr(self, 'parameters', {}).get(
            'cache_min_compute_seconds', 0.001)
        try:
            _floor = float(_floor)
        except (TypeError, ValueError):
            _floor = 0.001
        if _floor > 0 and (_time.monotonic() - _t0) < _floor:
            return res

        # Disk floor (her ruling c9, 2026-08-13). MarketSim._disk_check sets
        # this when free space falls below the floor. Skipping the WRITE is
        # safe in a way that skipping the READ would not be: the evaluation
        # sandbox is deterministic from (chain, scenario seed), so a chain
        # met again simply recomputes the identical value. The run pays time
        # and the disk stops growing under it.
        if getattr(self, '_cache_writes_suspended', False):
            return res

        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        # tmpfile per-pid so concurrent writers don't collide while
        # serializing; os.replace makes the final swap atomic.
        tmp = cachefile + f'.tmp.{os.getpid()}'
        try:
            with open(tmp, 'wb') as h:
                pickle.dump(res, h)
            os.replace(tmp, cachefile)
            # Cache-value law (her order, 2026-08-18): the sidecar the
            # pruner scores by — key, compute seconds, size.
            try:
                import json as _json
                with open(pickle_dir.rstrip('/').rsplit('/', 1)[0]
                          + '/cache_meta.jsonl', 'a') as _mf:
                    _mf.write(_json.dumps(dict(
                        key=pickle_name,
                        seconds=round(_time.monotonic() - _t0, 4),
                        bytes=os.path.getsize(cachefile))) + '\n')
            except OSError:
                pass
            # Keep self.pickles up-to-date for legacy code paths that
            # consult it (the MarketSim startup writes index.p; harmless
            # to also populate, though no longer required).
            if hasattr(self, 'pickles'):
                self.pickles[key] = pickle_name
        except MemoryError:
            # Result too large to serialize under the memcap: skip caching,
            # return it this once; the chain stays evaluable (and will
            # likely score None upstream anyway).
            try:
                os.unlink(tmp)
            except OSError:
                pass
        except AttributeError as e:
            # Match legacy error reporting for non-picklable results
            print(f": dumping {res} to {pickle_name} with error {e}")
            try:
                os.unlink(tmp)
            except OSError:
                pass

        return res
    return wrapped
