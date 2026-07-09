"""The NaN that exists only in the backward pass.

BUG: a hand-rolled distance ``sqrt(c**2 + x**2)`` is perfectly finite forward
— but its derivative is ``c / sqrt(...)``, which at (0, 0) is 0/0 = NaN. The
NaN is BORN in the backward pass; the forward computation never contains it.
(The historical fix is exactly why ``jnp.hypot`` now ships a guarded VJP.)
HARD TO DETECT: forward diagnostics — including forward taps — are clean by
construction. The NaN appears only when you differentiate, with no address
attached: ``grad`` just returns NaN.
THE HONEST BOUNDARY: taps riding along a grad transform observe the FORWARD
pass only (the documented contract) — jax-tap cannot see into the backward
pass of a program it is riding. BUT ``jax.grad(loss)`` is itself just a
function whose jaxpr contains the backward pass as ordinary primitives — so
tap THE DIFFERENTIATED FUNCTION, and the NaN's birth site gets an address.

Run:  uv run python demo/backward_pass_vjp_nan.py
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr

import jax
import jax.numpy as jnp

import jaxtap as tap


def make_loss():
    def body(c, x):
        # ╔═ jax-tap virtual injection ════════════════════════════════════╗
        # ║ forward taps here see r — always finite. The 0/0 lives in this  ║
        # ║ line's DERIVATIVE (c/r), which only exists inside grad(loss).   ║
        # ╚═ tap grad(loss) itself to give that NaN an address ══════════════╝
        r = jnp.sqrt(c**2 + x**2)  # <-- BUG LIVES HERE (in the backward pass)
        return r, None

    def loss(theta):
        c, _ = jax.lax.scan(body, theta, jnp.array([0.0, 3.0, 4.0]))
        return c

    return loss


def main() -> None:
    loss = make_loss()
    theta = jnp.float32(0.0)

    # ---------------- the bug, plainly ----------------
    fwd = float(loss(theta))
    grd = float(jax.grad(loss)(theta))
    print(f"forward: {fwd}   grad: {grd}   <- finite forward, NaN gradient, no address")

    # ---------------- act 1: the honest boundary ----------------
    buf = io.StringIO()
    with redirect_stderr(buf):
        watched = tap.verbose(loss, on_step=lambda e: None,
                              taps=[tap.watch_nan("sqrt"), tap.watch_nan("mul")])
        jax.block_until_ready(watched(theta))
    fwd_silent = "FAIL" not in buf.getvalue()
    print(f"\nact 1 — forward taps on loss():        "
          f"{'SILENT (forward is clean; the tap CANNOT see this bug)' if fwd_silent else 'fired?!'}")

    # ---------------- act 2: the escape hatch ----------------
    # grad(loss) is just a function; its jaxpr contains the backward pass as
    # ordinary primitives. Tap the division at the 0/0 site:
    caught = io.StringIO()
    with redirect_stderr(caught):
        jax.block_until_ready(tap.verbose(jax.grad(loss), on_step=lambda e: None,
                                          taps=[tap.watch_nan("div", once=True)])(theta))
    line = next((l for l in caught.getvalue().splitlines() if "FAIL" in l), "")
    print(f"act 2 — tap the DIFFERENTIATED function:\n  {line}")
    print("  -> the backward NaN, at its birth site, with an address and step.")

    ok = fwd_silent and (grd != grd) and "div" in line and "scan[0]" in line
    print(f"\nRESULT: boundary proven (forward taps blind) AND escape hatch works "
          f"(tap grad(f) itself) [{'PASS' if ok else 'FAIL'}]")


if __name__ == "__main__":
    main()
