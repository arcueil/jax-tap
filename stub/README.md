# jaxtap (alias package)

This is an alias/anti-squat package for **[jax-tap](https://github.com/arcueil/jax-tap)**
(pronounced "just tap") — zero-code-change runtime telemetry for JAX control flow.

Installing `jaxtap` installs `jax-tap`; the import name is `jaxtap` either way:

```bash
pip install jaxtap   # equivalent to: pip install jax-tap
```

```python
import jaxtap as tap
```

This package intentionally contains no code — it exists so that the PyPI name
matching the import name resolves to the real project. See the
[jax-tap repository](https://github.com/arcueil/jax-tap) for documentation.
