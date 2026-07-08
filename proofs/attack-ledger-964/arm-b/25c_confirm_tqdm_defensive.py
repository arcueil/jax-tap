"""Quick follow-up: confirm WHY 25b showed 0 uncaught exceptions -- does
tqdm itself swallow write-to-closed-stream errors internally?"""
import io
from tqdm.auto import tqdm

buf = io.StringIO()
bar = tqdm(total=10, file=buf, desc="probe")
bar.n = 1
bar.refresh()
buf.close()
try:
    bar.n = 2
    bar.refresh()
    print("tqdm.refresh() on a closed stream: no exception raised (tqdm is defensive)")
except Exception as e:
    print("tqdm.refresh() on a closed stream RAISED:", type(e).__name__, e)
try:
    bar.close()
    print("tqdm.close() on a closed stream: no exception raised")
except Exception as e:
    print("tqdm.close() on a closed stream RAISED:", type(e).__name__, e)
