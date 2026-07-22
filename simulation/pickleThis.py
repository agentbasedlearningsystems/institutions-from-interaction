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
present on the SnetSim instance (for backward compatibility with the
SnetSim init code that loads ``index.p``) but they are no longer
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


# Env vars whose value affects the simulation's output and therefore
# must be part of the cache key. Setting one of these to a non-default
# value (e.g. SNETSIM_VECTORSPACE_MAX_ROWS=10000) needs to produce
# DIFFERENT cache files than the unset-default behavior, otherwise we
# silently mix incompatible cached results.
#
# Backward compatibility note: if NONE of these env vars are set, we
# fall back to the original hash payload (just str(key)) so that
# pickles written by the pre-2026-05-21 code remain reachable. Pickles
# written with one of these vars explicitly set get a different hash
# even if the value matches the historical default.
_CACHE_NAMESPACE_ENV_VARS = (
    "SNETSIM_VECTORSPACE_MAX_ROWS",   # rows passed to clusterers
    "GCON_DATA_NROWS",                # rows loaded from CSV corpora
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
    extras = []
    for var in _CACHE_NAMESPACE_ENV_VARS:
        v = os.environ.get(var)
        if v is not None:
            extras.append(f"{var}={v}")
    if extras:
        payload = str(key) + "|" + ";".join(extras)
    else:
        payload = str(key)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32] + ".p"


def pickleThis(fn):  # decorator
    def wrapped(self, *args, **kwargs):
        # No key, nothing to cache — just call through.
        if not args:
            return fn(self, *args, **kwargs)
        key = args[0]
        pickle_name = _key_to_filename(key)
        pickle_dir = self.parameters['output_path'] + 'pickles/'
        cachefile = pickle_dir + pickle_name

        # Cache hit: read and return
        if os.path.exists(cachefile):
            try:
                with open(cachefile, 'rb') as h:
                    res = pickle.load(h)
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
                # too large to load under SNETSIM_EVAL_MEMCAP_GB — recompute
                # below (the compute path's MemoryError is caught by
                # memoise_pickle and scored None).
                pass

        # Cache miss: compute, then write atomically (tmp + rename).
        res = fn(self, *args, **kwargs)

        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir, exist_ok=True)
        # tmpfile per-pid so concurrent writers don't collide while
        # serializing; os.replace makes the final swap atomic.
        tmp = cachefile + f'.tmp.{os.getpid()}'
        try:
            with open(tmp, 'wb') as h:
                pickle.dump(res, h)
            os.replace(tmp, cachefile)
            # Keep self.pickles up-to-date for legacy code paths that
            # consult it (the SnetSim startup writes index.p; harmless
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
