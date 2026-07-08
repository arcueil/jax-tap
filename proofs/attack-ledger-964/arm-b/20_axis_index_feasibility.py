"""AYS (d) part 1: is 'gate the callback to fire only on axis_index==0'
actually implementable inside `_patched_scan`/`_inject_progress`, which have
NO access to the caller's mesh or axis names? Try the direct approaches and
see which fail.
"""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import jax.numpy as jnp

print("--- Attempt 1: jax.lax.axis_index() with no name, outside any explicit axis context ---")
try:
    idx = jax.lax.axis_index(None)
    print("succeeded:", idx)
except Exception as e:
    print("FAILED:", type(e).__name__, e)

print()
print("--- Attempt 2: is there a public API to enumerate the CURRENTLY active axis names "
      "at trace time, without the caller telling us? ---")
candidates = [
    "jax.core.get_axis_env",
    "jax.interpreters.pxla.thread_resources",
    "jax.core.thread_local_state",
]
for c in candidates:
    mod_path, _, attr = c.rpartition(".")
    try:
        import importlib
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        print(f"{c}: EXISTS ->", val)
    except Exception as e:
        print(f"{c}: NOT AVAILABLE ({type(e).__name__}: {e})")

print()
print("--- Attempt 3: jax.experimental.shard_map internals -- can we query the "
      "currently-open shard_map's mesh from inside a callee with zero cooperation "
      "from the caller? ---")
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import Mesh

mesh = Mesh(jax.devices(), axis_names=("my_custom_axis_name",))

def probe(x):
    # _patched_scan has NO idea the axis is called "my_custom_axis_name" --
    # it would need the caller to tell it, defeating the point of an
    # automatic, zero-config progress bar.
    idx = jax.lax.axis_index("my_custom_axis_name")
    return jnp.reshape(idx, (1,))

out = shard_map(probe, mesh=mesh, in_specs=P("my_custom_axis_name"),
                 out_specs=P("my_custom_axis_name"))(jnp.zeros(2))
print("axis_index WORKS when the exact axis name is hardcoded/known:", out)
print("but _patched_scan cannot know 'my_custom_axis_name' in general -- "
      "it is caller-chosen and never passed to jax.lax.scan or jax.debug.callback")

print()
print("CONCLUSION: axis-index gating requires the axis NAME, which is caller-defined "
      "and not observable from inside a generic jax.lax.scan monkeypatch. Not "
      "implementable as a zero-config library-internal fix without a new required "
      "parameter (e.g. progress_bar(shard_axis_name=...)) that reintroduces the "
      "per-callsite configuration the whole design is trying to avoid.")
