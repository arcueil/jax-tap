"""ARM L / Finding: __exit__ without a matching __enter__ (or a double __exit__)
emits a BOGUS 'foreign patch detected' warning and POISONS the global warn-once
flag, silencing a genuinely important future clobber warning.

Root cause: __exit__ guards the pop with `self._key is not None`, but the
restore/clobber-detection block runs whenever the registry is empty -- with NO
guard that THIS context ever entered.  When scan is at its original value (never
patched), `jax.lax.scan is _patched_scan` is False -> it takes the elif branch
and warns 'foreign patch replaced scan', setting _clobber_scan_warned=True.
"""
import warnings

import jax
import jaxtap as tap
import jaxtap._ashell as A


def clean():
    A._context_registry.clear()
    jax.lax.scan = A._original_scan
    jax.lax.while_loop = A._original_while
    A._session_scan = None
    A._session_while = None
    A._clobber_scan_warned = False
    A._clobber_while_warned = False


print("=== (A) __exit__ on a never-entered context ===")
clean()
ctx = tap.record()  # never entered; self._key is None, scan is original
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    ctx.__exit__(None, None, None)
msgs = [str(x.message) for x in w if "jaxtap" in str(x.message)]
print("warnings emitted:", msgs or "none")
print("scan clobber-warn flag poisoned to True?", A._clobber_scan_warned)
print("scan still original (not corrupted)?", jax.lax.scan is A._original_scan)

print()
print("=== (B) double __exit__ after a clean `with` block ===")
clean()
with tap.record():
    pass
# registry empty, scan original, self._key already None
with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter("always")
    # grab the context object to double-exit it
    c2 = tap.record()
    with c2:
        pass
    c2.__exit__(None, None, None)  # second exit
msgs2 = [str(x.message) for x in w2 if "jaxtap" in str(x.message)]
print("warnings on double-exit:", msgs2 or "none")

print()
print("=== consequence: poisoned flag silences a REAL later clobber ===")
clean()
# poison via bogus exit-without-enter
tap.record().__exit__(None, None, None)
poisoned = A._clobber_scan_warned
# now a genuine clobber-over-us happens
with warnings.catch_warnings(record=True) as w3:
    warnings.simplefilter("always")
    with tap.record():
        jax.lax.scan = lambda *a, **k: A._original_scan(*a, **k)  # real foreign clobber
    jax.lax.scan = A._original_scan
real_clobber_warned = any("jaxtap" in str(x.message) for x in w3)
print("flag poisoned by bogus exit?", poisoned)
print("genuine later clobber warned?", real_clobber_warned)
print(">>> BUG: real clobber went SILENT because a bogus exit poisoned the flag"
      if (poisoned and not real_clobber_warned) else ">>> not reproduced")
clean()
