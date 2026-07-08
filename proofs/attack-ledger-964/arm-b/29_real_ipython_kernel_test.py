"""AYS (3): real IPython kernel test (ephemeral overlay env via
`uv run --with ipykernel --with jupyter_client`, run against the blackjax
project's own resolved environment so blackjax/jax are importable inside
the spawned kernel -- no pyproject.toml/uv.lock changes, this is a
temporary runtime overlay only).

Sequence of REAL executed cells in one live kernel session:
  Cell 1: import blackjax/jax, capture _original_scan, enter a context
          WITHOUT `with` (cm.__enter__()), print patched-state.
  Cell 2..6: five totally UNRELATED cells (arithmetic, a gc.collect(), a
             list comprehension) -- simulating the user moving on with
             their notebook after forgetting to call __exit__.
  Cell 7: explicit `del cm` + gc.collect() -- does it heal in a REAL kernel,
          same as the flat-script simulation in round 1?

This settles: does the leaked patch survive across multiple real,
independent cell executions in a live IPython kernel (whole-session
persistence), or does IPython's own internals (traceback clearing, cell
namespace churn) heal it sooner (cell-or-two persistence)?
"""
import queue
import sys

from jupyter_client import KernelManager

km = KernelManager(kernel_name="python3")
km.start_kernel()
kc = km.client()
kc.start_channels()
kc.wait_for_ready(timeout=60)


def run_cell(code, timeout=30):
    msg_id = kc.execute(code)
    texts = []
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=timeout)
        except queue.Empty:
            texts.append("<TIMEOUT waiting for output>")
            break
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        mtype = msg["msg_type"]
        content = msg["content"]
        if mtype == "stream":
            texts.append(content["text"])
        elif mtype in ("execute_result", "display_data"):
            texts.append(content.get("data", {}).get("text/plain", ""))
        elif mtype == "error":
            texts.append("ERROR: " + "\n".join(content["traceback"]))
        elif mtype == "status" and content.get("execution_state") == "idle":
            break
    return "".join(texts)


try:
    print("--- sanity: blackjax importable inside the real kernel? ---")
    print(run_cell("import blackjax, jax; print('blackjax', blackjax.__file__)"))

    print("--- CELL 1: enter without `with`, no exception this time (the "
          "plainest, most realistic 'forgot to call __exit__' pattern) ---")
    print(run_cell(
        "from blackjax.progress_bar import _original_scan\n"
        "cm = blackjax.progress_bar(label='leaked-in-real-kernel')\n"
        "cm.__enter__()\n"
        "print('patched:', jax.lax.scan is not _original_scan)\n"
    ))

    for i in range(2, 7):
        print(f"--- CELL {i}: unrelated user work ---")
        print(run_cell(f"x{i} = sum(range({i} * 100)); print('cell {i} unrelated result:', x{i})"))

    print("--- CELL: explicit gc.collect() with `cm` still referenced "
          "(no del) -- does patch survive a REAL kernel's cyclic GC pass? ---")
    print(run_cell(
        "import gc; gc.collect()\n"
        "print('still patched after gc.collect() (cm still referenced):', "
        "jax.lax.scan is not _original_scan)\n"
    ))

    print("--- CELL 7: does the patch persist across 5 unrelated real cells "
          "in a live kernel? ---")
    print(run_cell(
        "print('patched after 5 unrelated cells:', jax.lax.scan is not _original_scan)\n"
    ))

    print("--- CELL 8: NOW explicitly `del cm` + gc.collect() -- does it heal? ---")
    print(run_cell(
        "del cm\n"
        "import gc; gc.collect()\n"
        "print('patched after del cm + gc.collect():', jax.lax.scan is not _original_scan)\n"
    ))
finally:
    kc.stop_channels()
    km.shutdown_kernel(now=True)
