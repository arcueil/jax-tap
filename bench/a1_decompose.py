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
bench/a1_decompose.py — Decompose the A1 mitigation per-callback overhead.

AYS-1 response: three micro-experiments to attribute the +13 µs/iter overhead.

Baseline (B0): while_loop, carry tap ships step + DIM leaves (no A1 arg).
(a) +1 dummy scalar: ships step + DIM leaves + active bool; host uses args[-1]
    via indexing (no star-unpack list, no .item() call) — isolates arg-shipping.
(b) +.item() call: ships step + DIM leaves + active bool; host uses args[-1].item()
    via index (no star-unpack list) — adds .item() cost on top of (a).
(c) full unpack: ships step + DIM leaves + active bool; host does
    ``*leaves, active_ = leaves_and_active`` + ``tuple(leaves)`` — current A1 impl.
(d) active-FIRST restructure: ships step + active + DIM leaves; host uses named
    positional args ``(step_, active_, *leaves)`` — no list allocation.

All arms use a while_loop with a DIM=8 float32 carry + int32 counter.
Trip count fixed at N=2000; K=25 repeats for stable medians.

Usage
-----
  uv run python bench/a1_decompose.py            # full run (N=2000, K=25)
  uv run python bench/a1_decompose.py --smoke    # N=200, K=7 (quick check)
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Any

import jax
import jax.lax as lax
import jax.numpy as jnp

DIM = 8
SEED = 42


# ---------------------------------------------------------------------------
# While-loop body shared across all arms
# ---------------------------------------------------------------------------


def _make_init(N: int) -> tuple[Any, Any]:
    key = jax.random.PRNGKey(SEED)
    v0 = jax.random.normal(key, (DIM,), dtype=jnp.float32)
    return v0, jnp.int32(0)


def _cond(carry: tuple) -> Any:
    _, counter = carry
    return counter < carry[2]  # carry = (v, counter, N_limit)


def _body_trivial(carry: tuple) -> tuple:
    v, counter, n = carry
    v2 = v * jnp.float32(1.001) + jnp.sin(v) * jnp.float32(0.001)
    return (v2, counter + 1, n)


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def warmup_and_time(jit_fn: Any, init: Any, N: int, K: int) -> tuple[float, float]:
    """1 warmup call then K timed repeats. Returns (median µs/iter, min µs/iter)."""
    jax.block_until_ready(jit_fn(init))
    times = []
    for _ in range(K):
        t0 = time.perf_counter()
        jax.block_until_ready(jit_fn(init))
        times.append(time.perf_counter() - t0)
    return (
        statistics.median(times) / N * 1e6,
        min(times) / N * 1e6,
    )


# ---------------------------------------------------------------------------
# Arm factories
# ---------------------------------------------------------------------------


def arm_b0_no_tap(N: int) -> tuple[Any, Any]:
    """B0 — bare while_loop, no tap at all."""
    n_arr = jnp.int32(N)
    init_v, init_c = _make_init(N)
    init = (init_v, init_c, n_arr)

    def body(carry: tuple) -> tuple:
        v, counter, n = carry
        v2 = v * jnp.float32(1.001) + jnp.sin(v) * jnp.float32(0.001)
        return (v2, counter + 1, n)

    def cond(carry: tuple) -> Any:
        _, counter, n = carry
        return counter < n

    def f(carry: tuple) -> Any:
        return lax.while_loop(cond, body, carry)

    return jax.jit(f), init


def arm_b1_baseline_tap(N: int) -> tuple[Any, Any]:
    """B1 — while_loop + tap (no A1): ships step + DIM leaves to host."""
    n_arr = jnp.int32(N)
    init_v, init_c = _make_init(N)
    init = (init_v, init_c, n_arr)

    def _host_b1(step_: Any, *leaves: Any) -> None:
        # Baseline pattern: step + DIM leaves, no active arg.
        # Mirrors verbose() before A1.
        pass  # noop on_step

    def cond(carry: tuple) -> Any:
        _, counter, n = carry
        return counter < n

    def body(carry: tuple) -> tuple:
        v, counter, n = carry
        v2 = v * jnp.float32(1.001) + jnp.sin(v) * jnp.float32(0.001)
        # Ship: step (int32) + DIM carry leaves (float32 each).
        jax.debug.callback(
            _host_b1, counter, *jax.tree_util.tree_leaves(v2), ordered=False
        )
        return (v2, counter + 1, n)

    def f(carry: tuple) -> Any:
        return lax.while_loop(cond, body, carry)

    return jax.jit(f), init


def arm_a_ship_only(N: int) -> tuple[Any, Any]:
    """(a) Ship dummy extra scalar (active bool); host accesses via index, no .item()."""
    n_arr = jnp.int32(N)
    init_v, init_c = _make_init(N)
    init = (init_v, init_c, n_arr)

    def _host_a(step_: Any, *args: Any) -> None:
        # args = (leaf0, ..., leaf_{DIM-1}, active_bool)
        # Access active via index (no star-unpack list), no .item() call.
        _ignored = args[-1]  # materialize the ref, no Python work beyond indexing
        # No TapEvent construction or on_step call (noop — isolate shipping cost).

    def cond(carry: tuple) -> Any:
        _, counter, n = carry
        return counter < n

    def body(carry: tuple) -> tuple:
        v, counter, n = carry
        v2 = v * jnp.float32(1.001) + jnp.sin(v) * jnp.float32(0.001)
        active_dummy = jnp.bool_(True)
        jax.debug.callback(
            _host_a,
            counter,
            *jax.tree_util.tree_leaves(v2),
            active_dummy,
            ordered=False,
        )
        return (v2, counter + 1, n)

    def f(carry: tuple) -> Any:
        return lax.while_loop(cond, body, carry)

    return jax.jit(f), init


def arm_b_item_only(N: int) -> tuple[Any, Any]:
    """(b) Ship + .item() call; index-based access (no star-unpack list)."""
    n_arr = jnp.int32(N)
    init_v, init_c = _make_init(N)
    init = (init_v, init_c, n_arr)

    def _host_b(step_: Any, *args: Any) -> None:
        # args = (leaf0, ..., leaf_{DIM-1}, active_bool)
        # Index-based: no list allocation.
        active_ = args[-1]
        if not active_.item():  # .item() call added vs (a)
            return

    def cond(carry: tuple) -> Any:
        _, counter, n = carry
        return counter < n

    def body(carry: tuple) -> tuple:
        v, counter, n = carry
        v2 = v * jnp.float32(1.001) + jnp.sin(v) * jnp.float32(0.001)
        active_dummy = jnp.bool_(True)
        jax.debug.callback(
            _host_b,
            counter,
            *jax.tree_util.tree_leaves(v2),
            active_dummy,
            ordered=False,
        )
        return (v2, counter + 1, n)

    def f(carry: tuple) -> Any:
        return lax.while_loop(cond, body, carry)

    return jax.jit(f), init


def arm_c_full_unpack(N: int) -> tuple[Any, Any]:
    """(c) Full *leaves, active_ = args + tuple(leaves) — current A1 implementation."""
    n_arr = jnp.int32(N)
    init_v, init_c = _make_init(N)
    init = (init_v, init_c, n_arr)

    def _host_c(step_: Any, *leaves_and_active: Any) -> None:
        # Current A1 pattern: star-unpack creates a Python list, then re-tuple.
        *leaves, active_ = leaves_and_active  # LIST allocation (DIM elements)
        if not active_.item():
            return
        _value = tuple(leaves)  # TUPLE allocation (converts list back)

    def cond(carry: tuple) -> Any:
        _, counter, n = carry
        return counter < n

    def body(carry: tuple) -> tuple:
        v, counter, n = carry
        v2 = v * jnp.float32(1.001) + jnp.sin(v) * jnp.float32(0.001)
        active_dummy = jnp.bool_(True)
        jax.debug.callback(
            _host_c,
            counter,
            *jax.tree_util.tree_leaves(v2),
            active_dummy,
            ordered=False,
        )
        return (v2, counter + 1, n)

    def f(carry: tuple) -> Any:
        return lax.while_loop(cond, body, carry)

    return jax.jit(f), init


def arm_d_active_first(N: int) -> tuple[Any, Any]:
    """(d) Restructured: active FIRST after step — no list allocation, leaves is a tuple."""
    n_arr = jnp.int32(N)
    init_v, init_c = _make_init(N)
    init = (init_v, init_c, n_arr)

    def _host_d(step_: Any, active_: Any, *leaves: Any) -> None:
        # active_ is a named positional arg — no unpacking.
        # leaves is already a tuple (Python *args) — no allocation.
        if not active_.item():
            return
        _value = leaves  # already a tuple, no conversion needed

    def cond(carry: tuple) -> Any:
        _, counter, n = carry
        return counter < n

    def body(carry: tuple) -> tuple:
        v, counter, n = carry
        v2 = v * jnp.float32(1.001) + jnp.sin(v) * jnp.float32(0.001)
        active_dummy = jnp.bool_(True)
        # active comes FIRST after step_, so leaves need no re-slicing.
        jax.debug.callback(
            _host_d,
            counter,
            active_dummy,
            *jax.tree_util.tree_leaves(v2),
            ordered=False,
        )
        return (v2, counter + 1, n)

    def f(carry: tuple) -> Any:
        return lax.while_loop(cond, body, carry)

    return jax.jit(f), init


def arm_e_sign_encode(N: int) -> tuple[Any, Any]:
    """(e) Sign-encode active into step — NO extra callback arg; same arg count as baseline.

    Encoding: active lane → step as-is (>=0); ghost lane → -(step+1) (<0).
    Host decodes: raw=step_.item(); if raw<0: return (ghost drop); step=raw.
    This eliminates the ~17 µs/iter arg-shipping overhead from (a).
    """
    n_arr = jnp.int32(N)
    init_v, init_c = _make_init(N)
    init = (init_v, init_c, n_arr)

    def _host_e(step_: Any, *leaves: Any) -> None:
        raw = step_.item()
        if raw < 0:
            return  # ghost lane — drop silently; no extra arg shipped
        _value = leaves  # already a tuple

    def cond(carry: tuple) -> Any:
        _, counter, n = carry
        return counter < n

    def body(carry: tuple) -> tuple:
        v, counter, n = carry
        v2 = v * jnp.float32(1.001) + jnp.sin(v) * jnp.float32(0.001)
        active_dummy = jnp.bool_(True)
        # Encode active into step: real → counter; ghost → -(counter+1).
        active_step = jax.lax.select(active_dummy, counter, -(counter + jnp.int32(1)))
        # Same arg count as baseline: step_ + DIM leaves (no extra bool).
        jax.debug.callback(
            _host_e, active_step, *jax.tree_util.tree_leaves(v2), ordered=False
        )
        return (v2, counter + 1, n)

    def f(carry: tuple) -> Any:
        return lax.while_loop(cond, body, carry)

    return jax.jit(f), init


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A1 callback-overhead decomposition bench"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke run: N=200, K=7 (quick check; results may be noisier)",
    )
    parsed = parser.parse_args()

    N = 200 if parsed.smoke else 2000
    K = 7 if parsed.smoke else 25
    smoke_tag = " *(smoke)*" if parsed.smoke else ""

    print(
        f"jax {jax.__version__} | device: {jax.devices()[0]}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"DIM={DIM} | N={N} | K={K}",
        file=sys.stderr,
        flush=True,
    )

    arms = [
        ("B0  no-tap", arm_b0_no_tap),
        ("B1  baseline-tap (no A1)", arm_b1_baseline_tap),
        ("(a) +1 dummy scalar, index-only", arm_a_ship_only),
        ("(b) +.item() call, index-based", arm_b_item_only),
        ("(c) full unpack (*leaves,active_)", arm_c_full_unpack),
        ("(d) active-first restructure", arm_d_active_first),
        ("(e) sign-encode (IMPL)", arm_e_sign_encode),
    ]

    rows = []
    for label, factory in arms:
        print(f"  timing {label!r}...", file=sys.stderr, flush=True)
        fn, init = factory(N)
        med, mn = warmup_and_time(fn, init, N, K)
        rows.append((label, med, mn))
        print(f"    {med:.2f} µs/iter", file=sys.stderr, flush=True)

    b0_med = rows[0][1]
    b1_med = rows[1][1]

    print()
    print(f"## A1 overhead decomposition — DIM={DIM}, N={N}, K={K}{smoke_tag}")
    print()
    print(
        "| arm | median µs/iter | min µs/iter | vs B0 (no-tap) | vs B1 (baseline-tap) |"
    )
    print("|-----|---------------|-------------|---------------|---------------------|")
    for label, med, mn in rows:
        vs_b0 = med - b0_med
        vs_b1 = med - b1_med
        vs_b0_s = f"{vs_b0:+.2f}" if label != "B0  no-tap" else "—"
        vs_b1_s = (
            f"{vs_b1:+.2f}"
            if label not in ("B0  no-tap", "B1  baseline-tap (no A1)")
            else "—"
        )
        print(f"| {label} | {med:.2f} | {mn:.2f} | {vs_b0_s} | {vs_b1_s} |")

    print()
    print("Delta breakdown (all vs B1 baseline-tap):")
    b1_label, b1_med, _ = rows[1]
    for label, med, _ in rows[2:]:
        delta = med - b1_med
        print(f"  {label}: {delta:+.2f} µs/iter")

    print()
    print("Notes:")
    print("  B0: no debug.callback at all — pure compute floor.")
    print("  B1: original tap (no A1 active arg) — pre-A1 baseline.")
    print("  (a)→(b) delta isolates .item() cost.")
    print("  (b)→(c) delta isolates *unpack + tuple() allocation cost.")
    print(
        "  (c)→(d) active-first restructure: marginal (dominant cost is arg-shipping, not unpack)."
    )
    print(
        "  (e) sign-encode: encodes active into step sign bit — same arg count as B1."
    )
    print("       Expected: close to B1 (no extra arg → no +17 µs shipping overhead).")


if __name__ == "__main__":
    main()
