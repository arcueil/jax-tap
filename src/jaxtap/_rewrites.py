"""
Scan and while_loop rewrite functions for the B-core walker.

Each rewrite takes the parsed eqn inputs, reconstructs the loop with an
augmented step-counter carry, fires ``jax.debug.callback`` per step, and
returns a flat list of output arrays matching the original eqn outvars.

The user body/cond functions never see the step counter — it is owned
entirely by the wrapper.  Taps fire via ``jax.debug.callback(..., ordered=False)``
which is vmap-safe (per-lane calls; ``ordered=True`` is not legal inside mapped
computation).

``sample_every`` gating lives one level up, in the ``tap_cb`` closure built by
``verbose()`` in ``__init__.py``.  That closure wraps the callback in a device-side
``lax.cond(step % k == 0, ...)`` before passing it here, so the rewrites always
call ``tap_cb`` unconditionally and correctness is preserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax.extend import core as jax_core

TapCallback = Callable[..., None]


def rewrite_scan(
    eqn: "jax_core.JaxprEqn",
    invals: list,
    tap_cb: TapCallback,
    ops: frozenset[str],
    here: str,
    interp_fn: Callable,
    sample_every: int = 1,
) -> list:
    """
    Rebuild a scan equation with a per-step counter in the carry and a
    ``tap_cb`` call after each body invocation.

    Parameters
    ----------
    eqn:
        The scan JaxprEqn being rewritten.
    invals:
        Flat list of live values for eqn.invars.
    tap_cb:
        Callback receiving ``(path, step, *carry_leaves)``.
    ops:
        Set of CF primitive names being tapped (forwarded to inner interpreter).
    here:
        Stable path string for this scan node.
    interp_fn:
        The recursive interpreter (``_interp``), to be called on sub-jaxprs.
    sample_every:
        Fire the tap only on steps 0, k, 2k, …  Implemented device-side via
        ``lax.cond`` so non-firing steps cross no host boundary.
    """
    p = eqn.params
    body: "jax_core.ClosedJaxpr" = p["jaxpr"]
    nc: int = p["num_consts"]
    ncar: int = p["num_carry"]

    consts = invals[:nc]
    init = invals[nc : nc + ncar]
    xs = invals[nc + ncar :]  # flat list; passed as pytree to scan

    # Forward all params except the ones we must reshape around.
    rest = {k: v for k, v in p.items() if k not in ("jaxpr", "num_consts", "num_carry")}

    k = jnp.int32(sample_every)

    def body_fn(carry_step, x):
        carry, step = carry_step
        # x arrives with the same pytree structure as xs.
        # Unpack into a flat list for the jaxpr body call.
        x_flat = list(x) if isinstance(x, (list, tuple)) else [x]
        outs = interp_fn(
            body.jaxpr,
            body.consts,
            [*consts, *carry, *x_flat],
            tap_cb,
            ops,
            here + "/",
        )
        new_carry = outs[:ncar]
        ys = outs[ncar:]
        # Gate the tap device-side: fire only when step % k == 0.
        # lax.cond is used so non-firing steps incur no host-boundary crossing.
        if sample_every == 1:
            # Avoid lax.cond overhead on the common path.
            tap_cb(here, step, *new_carry)
        else:
            jax.lax.cond(
                step % k == 0,
                lambda _: tap_cb(here, step, *new_carry),
                lambda _: None,
                None,
            )
        return (new_carry, step + 1), ys

    (carry_out, _), ys = jax.lax.scan(
        body_fn,
        (init, jnp.int32(0)),
        xs,
        **rest,
    )

    # Flatten ys back (scan returns same pytree structure as body returned).
    ys_flat = list(ys) if isinstance(ys, (list, tuple)) else [ys]
    return [*carry_out, *ys_flat]


def rewrite_while(
    eqn: "jax_core.JaxprEqn",
    invals: list,
    tap_cb: TapCallback,
    ops: frozenset[str],
    here: str,
    interp_fn: Callable,
    sample_every: int = 1,
) -> list:
    """
    Rebuild a while_loop equation with a step counter augmented into the carry
    and a ``tap_cb`` heartbeat after each body invocation.

    The cond function only sees the original carry (step counter hidden).

    sample_every:
        Fire the tap only on steps 0, k, 2k, …  Implemented device-side via
        ``lax.cond`` so non-firing steps cross no host boundary.
    """
    p = eqn.params
    cj: "jax_core.ClosedJaxpr" = p["cond_jaxpr"]
    cn: int = p["cond_nconsts"]
    bj: "jax_core.ClosedJaxpr" = p["body_jaxpr"]
    bn: int = p["body_nconsts"]

    cconsts = invals[:cn]
    bconsts = invals[cn : cn + bn]
    init = invals[cn + bn :]

    k = jnp.int32(sample_every)

    def cond_fn(carry_step):
        carry, _ = carry_step
        (pred,) = jax.core.eval_jaxpr(cj.jaxpr, cj.consts, *cconsts, *carry)
        return pred

    def body_fn(carry_step):
        carry, step = carry_step
        new_carry = interp_fn(bj.jaxpr, bj.consts, [*bconsts, *carry], tap_cb, ops, here + "/")
        if sample_every == 1:
            tap_cb(here, step, *new_carry)
        else:
            jax.lax.cond(
                step % k == 0,
                lambda _: tap_cb(here, step, *new_carry),
                lambda _: None,
                None,
            )
        return (new_carry, step + 1)

    carry_out, _ = jax.lax.while_loop(cond_fn, body_fn, (init, jnp.int32(0)))
    return list(carry_out)
