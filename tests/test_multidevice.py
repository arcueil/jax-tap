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
Multi-device behaviour: sharded jit, nested jit with explicit shardings,
shard_map and pmap across two devices.

Needs >= 2 devices on the default backend, so the module skips on a single
device.  Set JAXTAP_REQUIRE_MULTIDEVICE=1 to turn that skip into an error —
the GPU release gate does, so a single-GPU host cannot pass it silently.

Run on CPU with simulated devices:
    XLA_FLAGS=--xla_force_host_platform_device_count=2 uv run pytest tests/test_multidevice.py
Run on GPU:
    CUDA_VISIBLE_DEVICES=0,1 uv run pytest tests/test_multidevice.py
"""

from __future__ import annotations

import os

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import jaxtap as tap

N_DEV = 2
N_STEPS = 4
PER_DEV = 4
B = N_DEV * PER_DEV

if len(jax.devices()) < N_DEV:
    _msg = (
        f"needs >= {N_DEV} devices, found {len(jax.devices())} "
        f"on {jax.default_backend()}"
    )
    if os.environ.get("JAXTAP_REQUIRE_MULTIDEVICE") == "1":
        raise RuntimeError(_msg)
    pytest.skip(_msg, allow_module_level=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(c, x):
    return c * 0.5 + x, None


def _f(c0):
    carry, _ = lax.scan(_body, c0, jnp.arange(N_STEPS, dtype=jnp.float32))
    return carry


def _expected_carries(c0) -> np.ndarray:
    """(N_STEPS, len(c0)) array: the carry after each scan step."""
    c = np.asarray(c0, dtype=np.float32)
    out = []
    for t in range(N_STEPS):
        c = c * np.float32(0.5) + np.float32(t)
        out.append(c)
    return np.stack(out)


def _run(fn, *args):
    out = fn(*args)
    jax.block_until_ready(out)
    jax.effects_barrier()  # host callbacks are async: flush before reading events
    return out


def _step_values(events) -> list:
    """Sorted (step, carry) pairs; devices report in no fixed order."""
    return sorted((e.step, tuple(np.asarray(e.value[0]).tolist())) for e in events)


def _expected_per_device(c0) -> list:
    """Sorted (step, carry-shard) pairs: one per device per step."""
    carries = _expected_carries(c0)
    return sorted(
        (t, tuple(carries[t, d * PER_DEV : (d + 1) * PER_DEV].tolist()))
        for t in range(N_STEPS)
        for d in range(N_DEV)
    )


def _expected_global(c0) -> list:
    """Sorted (step, full carry) pairs: one per step."""
    carries = _expected_carries(c0)
    return [(t, tuple(carries[t].tolist())) for t in range(N_STEPS)]


@pytest.fixture(scope="module")
def mesh():
    return Mesh(np.array(jax.devices()[:N_DEV]), ("x",))


@pytest.fixture
def c0(mesh):
    return jax.device_put(jnp.arange(B, dtype=jnp.float32), NamedSharding(mesh, P("x")))


def _shard_map(mesh):
    return jax.shard_map(_f, mesh=mesh, in_specs=P("x"), out_specs=P("x"))


def _pmap():
    # Fresh wrapper each call: pmap caches on the wrapped function, and a
    # context-form run must trace (not hit a cache compiled outside it).
    return jax.pmap(lambda c: _f(c), devices=jax.devices()[:N_DEV])


# ---------------------------------------------------------------------------
# Setup sanity
# ---------------------------------------------------------------------------


def test_input_is_split_across_devices(c0):
    assert len({s.device for s in c0.addressable_shards}) == N_DEV


# ---------------------------------------------------------------------------
# jit over sharded inputs: one event per step, carrying the full value
# ---------------------------------------------------------------------------


def test_jit_sharded_b_form(c0):
    ref = _run(jax.jit(_f), c0)
    g, rec = tap.record(jax.jit(_f))
    out = _run(g, c0)

    np.testing.assert_array_equal(out, ref)
    assert out.sharding == ref.sharding
    assert _step_values(rec.events) == _expected_global(np.arange(B))


def test_jit_sharded_context_form(c0):
    ref = _run(jax.jit(_f), c0)
    with tap.record() as rec:
        out = _run(jax.jit(lambda c: _f(c)), c0)

    np.testing.assert_array_equal(out, ref)
    assert out.sharding == ref.sharding
    assert _step_values(rec.events) == _expected_global(np.arange(B))


@pytest.mark.parametrize(
    "out_spec",
    [
        P("x"),
        pytest.param(
            P(),
            marks=pytest.mark.xfail(
                strict=True,
                reason="known boundary: the jit re-wrap drops out_shardings, so a "
                "replicated inner result comes back sharded (values are identical)",
            ),
        ),
    ],
    ids=["sharded", "replicated"],
)
def test_nested_jit_explicit_shardings(mesh, c0, out_spec):
    # The walker re-wraps inner jits in a fresh jax.jit that does not carry
    # their in/out_shardings; the result must still match the untapped program.
    def make():
        inner = jax.jit(
            _f,
            in_shardings=NamedSharding(mesh, P("x")),
            out_shardings=NamedSharding(mesh, out_spec),
        )
        return jax.jit(lambda c: inner(c) * 2.0)

    ref = _run(make(), c0)
    g, rec = tap.record(make())
    out = _run(g, c0)

    np.testing.assert_array_equal(out, ref)
    assert out.sharding.is_equivalent_to(ref.sharding, out.ndim)
    assert _step_values(rec.events) == _expected_global(np.arange(B))


# ---------------------------------------------------------------------------
# shard_map / pmap: each device runs its own scan
# ---------------------------------------------------------------------------


def test_shard_map_context_form_one_event_per_device_per_step(mesh, c0):
    ref = _run(jax.jit(_shard_map(mesh)), c0)
    with tap.record() as rec:
        out = _run(jax.jit(_shard_map(mesh)), c0)

    np.testing.assert_array_equal(out, ref)
    assert _step_values(rec.events) == _expected_per_device(np.arange(B))


def test_pmap_context_form_one_event_per_device_per_step():
    x = jnp.arange(B, dtype=jnp.float32).reshape(N_DEV, PER_DEV)
    ref = _run(_pmap(), x)
    with tap.record() as rec:
        out = _run(_pmap(), x)

    np.testing.assert_array_equal(out, ref)
    assert _step_values(rec.events) == _expected_per_device(np.arange(B))


def test_shard_map_b_form_is_bitwise_identical(mesh, c0):
    ref = _run(jax.jit(_shard_map(mesh)), c0)
    g, _ = tap.record(jax.jit(_shard_map(mesh)))
    np.testing.assert_array_equal(_run(g, c0), ref)


def test_pmap_b_form_is_bitwise_identical():
    x = jnp.arange(B, dtype=jnp.float32).reshape(N_DEV, PER_DEV)
    ref = _run(_pmap(), x)
    g, _ = tap.record(_pmap())
    np.testing.assert_array_equal(_run(g, x), ref)


@pytest.mark.xfail(
    strict=True,
    reason="known boundary: the walker binds shard_map opaquely, so tap.record(f) "
    "sees no scans inside it (use the context form)",
)
def test_shard_map_b_form_emits_events(mesh, c0):
    g, rec = tap.record(jax.jit(_shard_map(mesh)))
    _run(g, c0)
    assert _step_values(rec.events) == _expected_per_device(np.arange(B))


@pytest.mark.xfail(
    strict=True,
    reason="known boundary: the walker binds pmap opaquely, so tap.record(f) "
    "sees no scans inside it (use the context form)",
)
def test_pmap_b_form_emits_events():
    x = jnp.arange(B, dtype=jnp.float32).reshape(N_DEV, PER_DEV)
    g, rec = tap.record(_pmap())
    _run(g, x)
    assert _step_values(rec.events) == _expected_per_device(np.arange(B))
