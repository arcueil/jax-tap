# Copyright 2026- The jax-tap Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

``emit_carry`` (M1d FIX 2)
--------------------------
When False, the carry-tap call (``tap_cb(here, step, *new_carry, ...)``) is
omitted from the reconstructed body.  The walker still descends into the body
jaxpr via ``interp_fn`` so that primitive taps and nested-loop carry taps can
fire — only THIS node's own carry heartbeat is suppressed.  This is a
Python-time (trace-time) decision; no device-side overhead is added when
``emit_carry=False``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

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
    outer_step: Any = None,
    *,
    emit_carry: bool = True,
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
    emit_carry:
        When False, skip the ``tap_cb(here, step, ...)`` carry-heartbeat call
        but still descend into the body via ``interp_fn`` so that primitive taps
        and nested-loop carry taps can fire.  Default: True.
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

    # total: the scan length, known at trace time.  Passed to tap_cb and
    # down into interp_fn so that primitive taps inside the body see the
    # correct TapEvent.total (the enclosing scan's length).
    total: int = p["length"]

    def body_fn(carry_step: Any, x: Any) -> Any:
        carry, step = carry_step
        # x arrives with the same pytree structure as xs.
        # Unpack into a flat list for the jaxpr body call.
        x_flat = list(x) if isinstance(x, (list, tuple)) else [x]
        # Pass the live step as the 7th argument, total as the 8th, and
        # True as the 9th (_in_loop override) so _interp gates primitive
        # taps with sample_every when inside this loop body.
        outs = interp_fn(
            body.jaxpr,
            body.consts,
            [*consts, *carry, *x_flat],
            tap_cb,
            ops,
            here + "/",
            step,
            total,
            True,  # _in_loop=True: we are now inside a scan body
        )
        new_carry = outs[:ncar]
        ys = outs[ncar:]
        # M1d FIX 2: emit_carry=False suppresses this node's carry heartbeat
        # while still having descended into the body above (for prim taps and
        # nested-loop carry taps).  emit_carry is a Python bool: no device overhead.
        if emit_carry:
            # tap_cb already has sample_every gating baked in (see verbose() in __init__.py).
            # total is a Python int captured from the enclosing rewrite_scan scope.
            tap_cb(here, step, *new_carry, total=total)
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
    outer_step: Any = None,
    *,
    emit_carry: bool = True,
) -> list:
    """
    Rebuild a while_loop equation with a step counter augmented into the carry
    and a ``tap_cb`` heartbeat after each body invocation.

    The cond function only sees the original carry (step counter hidden).

    ``sample_every`` gating lives one level up in ``verbose()`` — the rewrites
    always call ``tap_cb`` unconditionally.

    ``emit_carry``: when False, skip the ``tap_cb(here, step, ...)`` call but
    still descend into the body for primitive taps and nested-loop carry taps.
    """
    p = eqn.params
    cj: "jax_core.ClosedJaxpr" = p["cond_jaxpr"]
    cn: int = p["cond_nconsts"]
    bj: "jax_core.ClosedJaxpr" = p["body_jaxpr"]
    bn: int = p["body_nconsts"]

    cconsts = invals[:cn]
    bconsts = invals[cn : cn + bn]
    init = invals[cn + bn :]

    def cond_fn(carry_step: Any) -> Any:
        carry, _ = carry_step
        (pred,) = jax.core.eval_jaxpr(cj.jaxpr, cj.consts, *cconsts, *carry)
        return pred

    def body_fn(carry_step: Any) -> Any:
        carry, step = carry_step
        # A1 mitigation: evaluate the cond predicate on the PRE-BODY carry to
        # obtain a per-lane active mask.  Under plain (non-vmap) execution this
        # is always True — the body only runs when cond is True.  Under
        # vmap(while_loop), JAX runs max(trip_counts) joint iterations for ALL
        # lanes; lanes that have already finished execute the body with stale
        # carry values ("ghost iterations"), and active is False for those
        # lanes.  We pass active to tap_cb so the host-side _host closure can
        # drop ghost events before constructing a TapEvent.
        #
        # Cost: one extra cond_jaxpr evaluation per body iteration.  For
        # trivial conds (counter < N) this is free.  For expensive convergence-
        # check conds (norm(carry) > tol) it doubles the cond work per iteration
        # — measured in bench/while_cond_overhead.py; see known boundaries.
        (active,) = jax.core.eval_jaxpr(cj.jaxpr, cj.consts, *cconsts, *carry)
        # Pass the live step as the 7th argument, None as the 8th (total), and
        # True as the 9th (_in_loop override) so _interp gates primitive taps
        # with sample_every when inside this while body.
        new_carry = interp_fn(
            bj.jaxpr,
            bj.consts,
            [*bconsts, *carry],
            tap_cb,
            ops,
            here + "/",
            step,
            None,
            True,
        )
        # M1d FIX 2: emit_carry=False suppresses this node's carry heartbeat
        # while still having descended into the body above.
        if emit_carry:
            # tap_cb already has sample_every gating baked in (see verbose()).
            # total=None because while_loop length is unknown at trace time.
            # _while_active carries the per-lane active mask for ghost filtering.
            tap_cb(here, step, *new_carry, total=None, _while_active=active)
        return (new_carry, step + 1)

    carry_out, _ = jax.lax.while_loop(cond_fn, body_fn, (init, jnp.int32(0)))
    return list(carry_out)
