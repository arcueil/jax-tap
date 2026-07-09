import warnings
import numpy as np
import jax, jax.numpy as jnp
import jaxtap as tap

FAILS = []
def check(name, ok, d=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {d}")
    if not ok: FAILS.append(name)

orig_scan = jax.lax.scan
def f(x0):
    c, _ = jax.lax.scan(lambda c, x: (c + x, c), x0, jnp.arange(3.0))
    return c

# 1: emergency_restore heals a simulated stuck state
ctx = tap.record(); ctx.__enter__()
keeper = ctx  # keep alive so GC heal doesn't kick in
tap.emergency_restore()
check("emergency_restore restores originals", jax.lax.scan is orig_scan)
with tap.record() as rec:
    f(jnp.float32(0.0))
check("machinery healthy after emergency_restore", len(rec.events) == 3)

# 2: emergency_restore with FOREIGN patch on top -> warn, no clobber
with tap.record():
    foreign = lambda *a, **k: orig_scan(*a, **k)
    jax.lax.scan = foreign
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tap.emergency_restore()
    check("foreign-on-top: not clobbered", jax.lax.scan is foreign)
    check("foreign-on-top: warned", any("foreign" in str(x.message).lower() for x in w))
jax.lax.scan = orig_scan  # clean up for rest of probe

# 3: L5 guard exception-safety: user error inside verbose INSIDE a context;
#    context must still intercept afterwards (thread-local not left set)
def bad(x0):
    c, _ = jax.lax.scan(lambda c, x: (c + x[0], c), x0, jnp.arange(3.0))  # x[0] on scalar -> error
    return c
with tap.record() as rec2:
    try:
        tap.verbose(bad, on_step=lambda e: None)(jnp.float32(0.0))
    except Exception:
        pass
    f(jnp.float32(0.0))  # plain call: interception must still work
check("interception alive after verbose-raise inside ctx", len(rec2.events) == 3,
      f"({len(rec2.events)})")

# 4: fresh context restarts addressing at scan[0] (per-context counter)
with tap.record() as r1:
    f(jnp.float32(0.0)); f(jnp.float32(1.0))
with tap.record() as r2:
    f(jnp.float32(0.0))
p1 = sorted({e.path for e in r1.events}); p2 = sorted({e.path for e in r2.events})
check("ctx1 sequential unique", p1 == ["scan[0]", "scan[1]"], f"({p1})")
check("ctx2 restarts at scan[0]", p2 == ["scan[0]"], f"({p2})")

print("\n" + ("FIX-REVIEW R2: ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
