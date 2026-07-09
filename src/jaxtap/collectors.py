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
jaxtap collectors — host-side telemetry sinks.

All collectors are plain ``on_step``-compatible callables; none imports pandas
at module load so the core package imports with JAX only.

Classes
-------
FlightRecorder
    Accumulates TapEvents in memory; ``.df()`` exports a long-format pandas
    DataFrame.
JSONLWriter
    Writes one JSON object per TapEvent to a JSONL file; use as a context
    manager or close manually.

Functions
---------
read_jsonl(path) -> list[TapEvent]
    Reads a JSONL file written by :class:`JSONLWriter` and reconstructs a
    list of :class:`TapEvent` objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# FlightRecorder
# ---------------------------------------------------------------------------


class FlightRecorder:
    """
    On-step callable that accumulates :class:`TapEvent` objects in memory.

    Usage::

        recorder = FlightRecorder()
        tapped = tap.verbose(f, on_step=recorder)
        tapped(*args)
        df = recorder.df()

    **vmap note**: under ``jax.vmap``, a batched-carry tap fires once per
    vmap lane with no lane identifier.  ``.df()`` will contain N rows per
    ``(path, step)`` pair — one per lane.  Lane identity is NOT recorded
    (fabricating it would require a lane-index injection that jax-tap
    deliberately avoids for neutrality).
    """

    def __init__(self) -> None:
        self.events: list[Any] = []

    def __call__(self, event: Any) -> None:
        self.events.append(event)

    def df(self) -> "pd.DataFrame":
        """
        Return a long-format pandas DataFrame.

        Columns
        -------
        path, step
            Always present.
        value
            Present when ``TapEvent.value`` is a scalar or a 1-element
            sequence.
        value_0, value_1, …
            Present when the value is a multi-element sequence.
        <key>
            Present (one per key) when the value is a ``dict``.

        Raises
        ------
        ImportError
            When pandas is not installed.  Install with
            ``pip install jax-tap[pandas]`` or ``pip install pandas``.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for FlightRecorder.df(). "
                "Install it with: pip install jax-tap[pandas]  or  pip install pandas"
            ) from exc

        rows = []
        for event in self.events:
            row: dict[str, Any] = {"path": event.path, "step": int(event.step)}
            row.update(_value_to_columns(event.value))
            rows.append(row)

        return pd.DataFrame(rows)


def _value_to_columns(value: Any) -> dict[str, Any]:
    """Flatten a TapEvent value into a dict of column_name → Python scalar/list."""
    if isinstance(value, dict):
        return {str(k): _leaf_to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        leaves = list(value)
        if len(leaves) == 1:
            return {"value": _leaf_to_python(leaves[0])}
        return {f"value_{i}": _leaf_to_python(v) for i, v in enumerate(leaves)}
    # Scalar or single array
    return {"value": _leaf_to_python(value)}


def _leaf_to_python(v: Any) -> Any:
    """Convert a JAX/numpy array leaf to a JSON-serialisable Python object."""
    arr = np.asarray(v)
    if arr.ndim == 0:
        return arr.item()
    return arr.tolist()


# ---------------------------------------------------------------------------
# JSONLWriter / read_jsonl
# ---------------------------------------------------------------------------


class JSONLWriter:
    """
    On-step callable that writes one JSON object per :class:`TapEvent` to a
    JSONL file.

    Use as a context manager to ensure the file is closed::

        with JSONLWriter("log.jsonl") as w:
            tapped = tap.verbose(f, on_step=w)
            tapped(*args)

    Or close manually::

        w = JSONLWriter("log.jsonl")
        tapped = tap.verbose(f, on_step=w)
        tapped(*args)
        w.close()

    JSONL format (one JSON object per line)::

        {"path": "scan[0]", "step": 0, "value_kind": "tuple", "value": [1.0, 2.0]}
        {"path": "scan[0]", "step": 1, "value_kind": "scalar", "value": 3.5}
        {"path": "scan[0]", "step": 2, "value_kind": "dict", "value": {"a": 1.0}}

    ``value_kind`` encodes the original Python container type so
    :func:`read_jsonl` can reconstruct a value with the same structure.
    """

    def __init__(self, path: "str | Path") -> None:
        self._path = Path(path)
        self._file = self._path.open("w", encoding="utf-8")

    def __call__(self, event: Any) -> None:
        value, kind = _value_to_json(event.value)
        obj = {
            "path": event.path,
            "step": int(event.step),
            "value_kind": kind,
            "value": value,
        }
        self._file.write(json.dumps(obj, default=_json_default) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Flush and close the underlying file."""
        self._file.close()

    def __enter__(self) -> "JSONLWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _value_to_json(value: Any) -> "tuple[Any, str]":
    """Return (json_safe_value, kind_tag) for a TapEvent value."""
    if isinstance(value, dict):
        return {str(k): _leaf_to_python(v) for k, v in value.items()}, "dict"
    if isinstance(value, (list, tuple)):
        return [_leaf_to_python(v) for v in value], "tuple"
    return _leaf_to_python(value), "scalar"


def _json_default(obj: Any) -> Any:
    """Fallback JSON serialiser for numpy scalars."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__!r} is not JSON serialisable")


def read_jsonl(path: "str | Path") -> list:
    """
    Read a JSONL file written by :class:`JSONLWriter`.

    Returns a list of :class:`TapEvent` objects with values reconstructed as:

    - ``"tuple"`` kind → Python tuple of numpy scalars / arrays
    - ``"dict"`` kind → dict with numpy scalar / array values
    - ``"scalar"`` kind → numpy scalar

    Parameters
    ----------
    path:
        Path to the JSONL file.
    """
    from . import TapEvent  # import here to avoid circular dependency at module load

    events = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            kind = obj.get("value_kind", "tuple")
            raw = obj["value"]
            value = _reconstruct_value(raw, kind)
            events.append(TapEvent(path=obj["path"], step=obj["step"], value=value))
    return events


def _reconstruct_value(raw: Any, kind: str) -> Any:
    """Reconstruct a TapEvent value from its JSON representation."""
    if kind == "tuple":
        return tuple(_to_numpy(v) for v in raw)
    if kind == "dict":
        return {k: _to_numpy(v) for k, v in raw.items()}
    # scalar
    return _to_numpy(raw)


def _to_numpy(v: Any) -> Any:
    """Convert a JSON-parsed value (scalar or nested list) to numpy array."""
    arr = np.asarray(v)
    if arr.ndim == 0:
        return arr.item()
    return arr
