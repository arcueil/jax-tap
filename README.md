# jax-tap

Zero-code-change runtime telemetry for JAX control flow. Instrument `lax.scan` /
`lax.while_loop` with carry taps and heartbeats — the observed and production
programs are bitwise-identical; the lens is the only difference.

## Install

```
pip install jax-tap
```

## Quick start

```python
import jaxtap as tap

events = []

def on_step(event: tap.TapEvent) -> None:
    events.append(event)           # path, step, value

tapped_f = tap.verbose(f, on_step=on_step)
result = tapped_f(*args)           # bitwise-identical to f(*args)
```
