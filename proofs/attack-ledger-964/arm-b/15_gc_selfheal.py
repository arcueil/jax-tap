"""AYS (a): does dropping the last reference to `cm` (without ever calling
__exit__) trigger generator finalization -> finally block -> self-heal?
Test both: (1) plain refcounting (del cm, no gc.collect), (2) whether the
display thread gets joined too, (3) the realistic Jupyter-adjacent case
where an exception's traceback keeps the failing frame (and thus `cm`)
alive even after the enclosing scope exits.
"""
import gc
import sys
import threading

import jax
import jax.numpy as jnp

import blackjax
from blackjax.progress_bar import _original_scan, _progress_registry

print("=== Case 1: explicit `del cm`, pure refcounting, no gc.collect() ===")
cm = blackjax.progress_bar(label="gc-test")
state = cm.__enter__()
print("patched:", jax.lax.scan is not _original_scan)
print("registry size:", len(_progress_registry))
display_thread = state._display_thread
print("display thread alive:", display_thread.is_alive())

del cm
print("--- after `del cm` (no explicit gc.collect) ---")
print("patched still?:", jax.lax.scan is not _original_scan)
print("registry size:", len(_progress_registry))
print("display thread alive:", display_thread.is_alive())

gc.collect()
print("--- after explicit gc.collect() (belt and suspenders) ---")
print("patched still?:", jax.lax.scan is not _original_scan)
print("registry size:", len(_progress_registry))
print("display thread alive:", display_thread.is_alive())

print()
print("=== Case 2: function-local `cm` going out of scope on normal return ===")
def helper():
    local_cm = blackjax.progress_bar(label="scope-test")
    local_cm.__enter__()
    return  # local_cm falls out of scope here, no exception

helper()
print("patched after helper() returns (no gc.collect):", jax.lax.scan is not _original_scan)
gc.collect()
print("patched after gc.collect():", jax.lax.scan is not _original_scan)
print("registry size:", len(_progress_registry))

print()
print("=== Case 3: exception raised, traceback retained (IPython-like) ===")
# IPython stores the last exception (and hence its traceback -> failing
# frame -> failing frame's locals, incl. `cm`) in sys.last_value /
# sys.last_traceback until the NEXT exception overwrites it, or explicit
# clearing. Simulate the same retention pattern with a plain `except`
# clause that keeps the exception object alive.
def cell_that_errors():
    cm2 = blackjax.progress_bar(label="traceback-retained")
    cm2.__enter__()
    raise RuntimeError("simulated cell error between __enter__ and __exit__")

retained_exc = None
try:
    cell_that_errors()
except RuntimeError as e:
    retained_exc = e  # mimics sys.last_value holding the exception
    sys.last_traceback = e.__traceback__  # mimics IPython's real behavior

print("patched while exception+traceback still referenced (no gc.collect):",
      jax.lax.scan is not _original_scan)
gc.collect()
print("patched after gc.collect() while exception STILL referenced:",
      jax.lax.scan is not _original_scan)
print("registry size while traceback retained:", len(_progress_registry))

# Now drop the retained exception/traceback (simulating the NEXT cell
# executing without error, or an explicit `del`/`%xdel`).
del retained_exc
del sys.last_traceback
gc.collect()
print("patched after dropping the retained traceback + gc.collect():",
      jax.lax.scan is not _original_scan)
print("registry size after dropping traceback:", len(_progress_registry))
