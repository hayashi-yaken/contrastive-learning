"""Tiny logging helper: timestamped, immediately flushed (so nohup/file
redirection shows progress in real time instead of buffering it)."""
import sys
import time

_START = time.time()


def log(msg):
    """Print an elapsed-time-stamped line and flush immediately."""
    elapsed = time.time() - _START
    print(f"[{elapsed:8.1f}s] {msg}", flush=True)


def reset_clock():
    global _START
    _START = time.time()
