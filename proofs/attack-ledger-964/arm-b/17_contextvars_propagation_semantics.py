"""AYS (b) continued: before proposing contextvars as the redesign, verify
Python's actual propagation semantics for raw threading.Thread vs
ThreadPoolExecutor.submit -- these determine exactly which of the two
'legitimate delegation' patterns from 16_worker_thread_delegation.py a
contextvars-based redesign would preserve vs regress.
"""
import concurrent.futures
import contextvars
import threading

cv = contextvars.ContextVar("test_var", default="DEFAULT-not-propagated")
cv.set("SET-ON-MAIN-THREAD")

def read_and_report(label):
    print(f"{label}: sees value =", repr(cv.get()))

print("main thread directly:", repr(cv.get()))

t = threading.Thread(target=read_and_report, args=("raw threading.Thread",))
t.start()
t.join()

with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
    fut = ex.submit(read_and_report, "ThreadPoolExecutor.submit")
    fut.result()
