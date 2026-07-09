# Copyright 2026 The jax-tap Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
M1c UX-fix tests: once=True, single-line tap.print.

Covers:
- watch_nan(once=True) fires exactly ONE FAIL line even when NaN spans many steps
- tap.print(once=True) fires exactly once
- once budget resets for a new verbose() call (fresh _once_fired set)
- tap.print emitted line contains no embedded newlines

Run with: uv run pytest tests/test_m1c_ux_fixes.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jaxtap as tap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NAN_N = 10  # scan length — must be a concrete Python int for jax.lax.scan


def _chol_nan_scan(x0):
    """Scan whose Cholesky goes NaN from step 2 onward (c → 1, matrix singular)."""

    def body(carry, _):
        k = carry
        c = 1.0 - 10.0 ** (-jnp.minimum(k, 12.0))
        M = jnp.array([[1.0, c], [c, 1.0]], dtype=c.dtype)
        L = jnp.linalg.cholesky(M)
        return carry + 1.0, jnp.sum(L)

    return jax.lax.scan(body, x0, None, length=_NAN_N)


def _sin_scan_3(x0):
    """3-step scan through sin."""

    def body(carry, _):
        return jnp.sin(carry), carry

    return jax.lax.scan(body, x0, None, length=3)


# ---------------------------------------------------------------------------
# once=True: fires exactly once
# ---------------------------------------------------------------------------


def test_watch_nan_once_fires_exactly_once(capsys):
    """watch_nan(once=True) emits exactly one FAIL line even when NaN spans many steps."""
    # float32 cholesky goes NaN from step 7 onward (multiple steps have NaN)
    tapped = tap.verbose(
        _chol_nan_scan,
        on_step=lambda e: None,
        taps=[tap.watch_nan("cholesky", once=True)],
    )
    result = tapped(jnp.float32(1.0))
    jax.block_until_ready(result)
    captured = capsys.readouterr()
    fail_lines = [line for line in captured.err.splitlines() if "FAIL" in line]
    assert len(fail_lines) == 1, (
        f"expected exactly 1 FAIL line, got {len(fail_lines)}: {fail_lines}"
    )


def test_watch_nan_once_false_fires_multiple(capsys):
    """Control: watch_nan(once=False, default) emits multiple FAIL lines."""
    tapped = tap.verbose(
        _chol_nan_scan,
        on_step=lambda e: None,
        taps=[tap.watch_nan("cholesky", once=False)],
    )
    result = tapped(jnp.float32(1.0))
    jax.block_until_ready(result)
    captured = capsys.readouterr()
    fail_lines = [line for line in captured.err.splitlines() if "FAIL" in line]
    # Multiple steps have NaN cholesky; once=False should yield >1 FAIL lines
    assert len(fail_lines) > 1, f"expected >1 FAIL lines with once=False, got {len(fail_lines)}"


def test_tap_print_once_fires_exactly_once(capsys):
    """tap.print(once=True) emits exactly one line even for a 3-step scan."""
    prim_counts = tap.primitives(_sin_scan_3, jnp.float32(1.0))
    sin_name = next((k for k in prim_counts if k == "sin"), None)
    assert sin_name is not None

    tapped = tap.verbose(
        _sin_scan_3,
        on_step=lambda e: None,
        taps=[tap.print(sin_name, once=True)],
    )
    result = tapped(jnp.float32(1.0))
    jax.block_until_ready(result)
    captured = capsys.readouterr()
    tap_lines = [line for line in captured.err.splitlines() if "[tap]" in line]
    assert len(tap_lines) == 1, f"expected 1 line with once=True, got {len(tap_lines)}"


def test_once_rearms_for_new_verbose_call(capsys):
    """once=True resets when verbose() is called again (fresh _once_fired per call)."""
    spec = tap.watch_nan("cholesky", once=True)

    for call_idx in range(2):
        # Each verbose() creates a new _once_fired; the spec's once budget resets.
        tapped = tap.verbose(
            _chol_nan_scan,
            on_step=lambda e: None,
            taps=[spec],
        )
        result = tapped(jnp.float32(1.0))
        jax.block_until_ready(result)

    captured = capsys.readouterr()
    fail_lines = [line for line in captured.err.splitlines() if "FAIL" in line]
    # Two verbose() calls × 1 FAIL each = exactly 2 FAIL lines total
    assert len(fail_lines) == 2, (
        f"expected 2 FAIL lines (one per verbose() call), got {len(fail_lines)}: {fail_lines}"
    )


# ---------------------------------------------------------------------------
# Single-line: tap.print output must not contain embedded newlines
# ---------------------------------------------------------------------------


def _large_matrix_scan(x0):
    """Scan that passes a 4×4 matrix through sin — numpy repr wraps across lines."""

    def body(carry, _):
        M = jnp.ones((4, 4), dtype=jnp.float32) * carry
        v = jnp.sin(M)
        return carry + 1.0, v.sum()

    return jax.lax.scan(body, x0, None, length=2)


def test_tap_print_no_embedded_newlines(capsys):
    """Each emitted line contains no embedded newlines — a 4×4 matrix must fit on one line."""
    prim_counts = tap.primitives(_large_matrix_scan, jnp.float32(1.0))
    sin_name = next((k for k in prim_counts if k == "sin"), None)
    assert sin_name is not None

    tapped = tap.verbose(
        _large_matrix_scan,
        on_step=lambda e: None,
        taps=[tap.print(sin_name)],
    )
    result = tapped(jnp.float32(1.0))
    jax.block_until_ready(result)
    captured = capsys.readouterr()

    tap_lines = [line for line in captured.err.splitlines() if "[tap]" in line]
    assert len(tap_lines) == 2, f"expected 2 tap lines, got {len(tap_lines)}"
    for line in tap_lines:
        # splitlines() already separated on \n; each entry must not contain further \n
        assert "\n" not in line, f"embedded newline in line: {line!r}"
        # The 4×4 repr has spaces but the event must be on ONE line
        assert line.startswith("[tap] "), f"unexpected prefix: {line!r}"
