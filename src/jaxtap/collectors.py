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
jaxtap collectors — host-side telemetry sinks.

This module provides callables compatible with the ``on_step`` parameter of
:func:`jaxtap.verbose` and :func:`jaxtap.record`. All collectors are plain
callables that accept :class:`jaxtap.TapEvent` objects.

No collector imports pandas at module load — the core package depends only on
JAX and numpy. The optional pandas dependency is imported only when
:meth:`FlightRecorder.df` is called.

Notes
-----
Use collectors with the B-form of :func:`jaxtap.record`::

    recorder = FlightRecorder()
    tapped_f = jaxtap.verbose(f, on_step=recorder)
    tapped_f(*args)
    df = recorder.df()

Or use JSONLWriter as a context manager for streaming output::

    with JSONLWriter("events.jsonl") as w:
        tapped_f = jaxtap.verbose(f, on_step=w)
        tapped_f(*args)
    events = read_jsonl("events.jsonl")
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

    This class acts as a plain callable compatible with the ``on_step``
    parameter of :func:`jaxtap.verbose` and :func:`jaxtap.record`. It stores
    every :class:`TapEvent` delivered to it in an internal list, accessible
    via the ``.events`` attribute or exported to a pandas DataFrame via
    :meth:`.df`.

    Attributes
    ----------
    events : list
        List of :class:`TapEvent` objects accumulated since instantiation
        (or since the last manual clear). Readable at any time; appended to
        as events fire.

    Notes
    -----
    **vmap behavior**: under ``jax.vmap``, a batched-carry tap fires once per
    vmap lane with no lane identifier. ``.df()`` will contain N rows per
    ``(path, step)`` pair — one per lane. Lane identity is NOT recorded
    (fabricating it would require a lane-index injection that jax-tap
    deliberately avoids for neutrality).

    Examples
    --------
    Accumulate events and export to pandas::

        import jaxtap as tap

        recorder = tap.FlightRecorder()
        tapped = tap.verbose(f, on_step=recorder)
        tapped(*args)

        df = recorder.df()  # pandas.DataFrame with columns: path, step, value, ...
        print(df)
    """

    def __init__(self) -> None:
        self.events: list[Any] = []

    def __call__(self, event: Any) -> None:
        self.events.append(event)

    def df(self) -> "pd.DataFrame":
        """
        Return a long-format pandas DataFrame of accumulated events.

        Exports all :class:`TapEvent` objects to a pandas DataFrame, with one
        row per event. The pytree structure of ``TapEvent.value`` is flattened
        into individual columns using :func:`jaxtap.collectors._value_to_columns`.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the following columns:

            - ``path`` : str — the event's stable address
            - ``step`` : int — the enclosing loop step (or -1 for primitives)
            - Value columns (structure-dependent):

              - ``value`` : present when ``TapEvent.value`` is a scalar or
                1-element sequence
              - ``value_0``, ``value_1``, … : present when the value is a
                multi-element sequence (e.g., tuple or list)
              - Individual keys (e.g., ``a``, ``b``) : present when the value
                is a dict

        Notes
        -----
        With mixed carry/output events, ``df()`` does not distinguish kind —
        filter ``[e for e in rec.events if e.kind == "output"]`` first if you
        need separation.

        Raises
        ------
        ImportError
            When pandas is not installed. Install with::

                pip install jax-tap[pandas]
                # or
                pip install pandas

        Examples
        --------
        Export events to a DataFrame for analysis::

            recorder = tap.FlightRecorder()
            tapped = tap.verbose(f, on_step=recorder)
            tapped(*args)

            df = recorder.df()
            print(df[["path", "step", "value"]])
            #           path  step     value
            # 0      scan[0]     0         1.5
            # 1      scan[0]     1         2.5
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
    On-step callable that writes one JSON object per :class:`TapEvent` to a file.

    Writes telemetry events to a JSONL file (one JSON object per line) for
    streaming or batch analysis. Each line is a self-contained JSON object
    with the event's path, step, and a JSON-safe representation of the value.

    The ``value_kind`` field encodes the original Python container type
    (scalar, tuple, or dict) so :func:`read_jsonl` can reconstruct the value
    with the same structure (see Notes below).

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the JSONL file. The file is created if it does not exist;
        if it exists, it is truncated.

    Notes
    -----
    **File I/O**: The file is opened in write mode at instantiation. Use as a
    context manager to ensure it is closed on exit, or call :meth:`.close`
    manually.

    **JSONL format**: Each line is a JSON object::

        {"path": "scan[0]", "step": 0, "value_kind": "tuple", "value": [1.0, 2.0], "kind": "carry"}
        {"path": "scan[0]", "step": 1, "value_kind": "scalar", "value": 3.5, "kind": "output"}
        {"path": "scan[0]", "step": 2, "value_kind": "dict", "value": {"a": 1.0}, "kind": "carry"}

    ``value_kind`` is one of ``"scalar"``, ``"tuple"``, or ``"dict"``.
    ``kind`` is the event kind: ``"carry"`` for carry-tap events or ``"output"``
    for y-tap events (added in 0.3.0).
    JAX/numpy arrays are converted to Python scalars or lists.

    Examples
    --------
    Write events to a file::

        with tap.JSONLWriter("events.jsonl") as w:
            tapped = tap.verbose(f, on_step=w)
            tapped(*args)

    Read the file back::

        events = tap.read_jsonl("events.jsonl")
        for event in events:
            print(f"{event.path} step {event.step}: {event.value}")
    """

    def __init__(self, path: "str | Path") -> None:
        self._path = Path(path)
        self._file = self._path.open("w", encoding="utf-8")

    def __call__(self, event: Any) -> None:
        value, value_kind = _value_to_json(event.value)
        obj = {
            "path": event.path,
            "step": int(event.step),
            "value_kind": value_kind,
            "value": value,
            # "kind" (event kind: "carry" | "output") is distinct from "value_kind"
            # (value container type: "scalar" | "tuple" | "dict").
            # Serialized as "kind" so y-tap events survive a JSONL round-trip.
            # getattr guard: tolerates TapEvent objects from code that predates
            # the kind field (defaults to "carry", the backward-compat value).
            "kind": getattr(event, "kind", "carry"),
        }
        self._file.write(json.dumps(obj, default=_json_default) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Flush and close the underlying JSONL file.

        This method flushes any pending writes and closes the file handle.
        After calling :meth:`.close`, the writer cannot be used to write
        more events.  The file is then ready for reading (e.g., with
        :func:`read_jsonl`).

        Calling :meth:`.close` more than once is safe (idempotent).
        """
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

    Parses each line as a JSON object, reconstructs :class:`TapEvent` objects,
    and returns them as a list. The ``value_kind`` field in each JSON object
    determines how to reconstruct the pytree structure of the ``value`` field.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the JSONL file (created by :class:`JSONLWriter`).

    Returns
    -------
    list
        A list of :class:`TapEvent` objects with values reconstructed as:

        - ``"tuple"`` value_kind → Python tuple of numpy scalars / arrays
        - ``"dict"`` value_kind → dict with numpy scalar / array values
        - ``"scalar"`` value_kind → numpy scalar

        ``TapEvent.kind`` (``"carry"`` or ``"output"``) is restored from the
        ``"kind"`` field in the JSONL record.  Files written before 0.3.0
        (which do not have a ``"kind"`` field) default to ``"carry"``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If a line in the file is not valid JSON.

    Examples
    --------
    Write events, then read them back::

        with tap.JSONLWriter("events.jsonl") as w:
            tapped = tap.verbose(f, on_step=w)
            tapped(*args)

        events = tap.read_jsonl("events.jsonl")
        for event in events:
            print(f"Path: {event.path}, Step: {event.step}, Value: {event.value}")
    """
    from . import TapEvent  # import here to avoid circular dependency at module load

    events = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            value_kind = obj.get("value_kind", "tuple")
            raw = obj["value"]
            value = _reconstruct_value(raw, value_kind)
            # "kind" is the event kind ("carry" | "output"), distinct from
            # "value_kind" (value container type).  Old files without "kind"
            # default to "carry" for backward compatibility.
            event_kind = obj.get("kind", "carry")
            events.append(
                TapEvent(
                    path=obj["path"],
                    step=obj["step"],
                    value=value,
                    kind=event_kind,
                )
            )
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
