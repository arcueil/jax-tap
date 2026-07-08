"""ARM L / Foreign-patch matrix edges beyond tests #8/#9:

(A) warn-once flags (_clobber_scan_warned / _clobber_while_warned) are GLOBAL and
    never reset -> a SECOND, genuinely independent foreign clobber later in the
    same process is SILENT (no warning). Observability bug.

(B) when a foreign patch is installed OVER us and we're the last ctx out, exit
    sets _session_scan=None -> any pre-session foreign patch that we had captured
    and chained to is forgotten. Show the state.
"""
import warnings

import jax
import jax.numpy as jnp
import jaxtap as tap
import jaxtap._ashell as A


def clean():
    A._context_registry.clear()
    jax.lax.scan = A._original_scan
    jax.lax.while_loop = A._original_while
    A._session_scan = None
    A._session_while = None


def make_foreign(tag, calls):
    def _f(f, init, xs=None, length=None, **kwargs):
        calls.append(tag)
        return A._original_scan(f, init, xs=xs, length=length, **kwargs)
    return _f


print("=== (A) second independent clobber is SILENT (warn-once never resets) ===")
clean()
A._clobber_scan_warned = False  # simulate fresh process

# First clobber event -> should warn.
with warnings.catch_warnings(record=True) as w1:
    warnings.simplefilter("always")
    with tap.record():
        jax.lax.scan = make_foreign("clobber1", [])  # patch over us
    # exit: sees non-patched scan -> warns
n_warn_1 = sum("jaxtap" in str(x.message) for x in w1)
print("first clobber warned?", n_warn_1 >= 1, f"({n_warn_1} warning[s])")
jax.lax.scan = A._original_scan  # cleanup

# Second, totally independent clobber event later in the same process.
with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter("always")
    with tap.record():
        jax.lax.scan = make_foreign("clobber2", [])  # patch over us again
    # exit: _clobber_scan_warned is still True from before -> SILENT
n_warn_2 = sum("jaxtap" in str(x.message) for x in w2)
print("second clobber warned?", n_warn_2 >= 1, f"({n_warn_2} warning[s])")
jax.lax.scan = A._original_scan
print(">>> BUG (observability): second real clobber is silent"
      if (n_warn_1 >= 1 and n_warn_2 == 0) else ">>> both warned / neither warned")

print()
print("=== (B) foreign-BEFORE chained, then foreign-OVER-us on exit drops the chain ===")
clean()
A._clobber_scan_warned = False
pre_calls = []
pre_foreign = make_foreign("pre", pre_calls)
jax.lax.scan = pre_foreign  # foreign installed BEFORE we enter

with warnings.catch_warnings(record=True):
    warnings.simplefilter("always")
    with tap.record():
        # session_scan should have captured pre_foreign here
        captured = A._session_scan
        # now a DIFFERENT foreign patch lands over us
        jax.lax.scan = make_foreign("over", [])
    # exit: not _patched_scan -> warn + leave 'over' in place; _session_scan reset to None
print("captured pre-foreign as _session_scan during ctx?", captured is pre_foreign)
print("after exit: scan is the 'over' foreign (not restored to pre)?",
      jax.lax.scan is not pre_foreign and jax.lax.scan is not A._original_scan)
print("_session_scan after exit (chain forgotten -> None):", A._session_scan)
clean()
