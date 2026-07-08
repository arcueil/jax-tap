"""Attack #8 (process hygiene): does merely importing blackjax patch jax.lax.scan?"""
import jax

pre = jax.lax.scan
import blackjax  # noqa: E402

post = jax.lax.scan
print("scan identical after import:", pre is post)
print("registry empty:", blackjax.progress_bar.__wrapped__ if hasattr(blackjax.progress_bar, "__wrapped__") else "n/a")
from blackjax.progress_bar import _progress_registry
print("registry contents:", _progress_registry)
