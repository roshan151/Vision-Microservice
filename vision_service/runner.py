"""Serialized inference with bounded waiting.

There is one GPU. Two inferences at once do not go faster, they go slower and
less predictably, so work is funnelled through a **single worker thread**. That
one worker is what serializes access; no additional lock is needed, and the
tricky failure mode of a lock released by a timeout while its thread still owns
the GPU cannot happen.

Requests that arrive while the worker is busy wait. Requests that arrive while
too many are already waiting are **refused with 503**, not queued — the server
half of PLAN.md standing rule #8. A refusal the Pi can react to immediately
beats an answer that arrives after the Pi has stopped caring.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Busy(RuntimeError):
    """Too many requests already waiting for the GPU."""


class InferenceTimeout(RuntimeError):
    """The worker did not finish within the timeout."""


class InferenceRunner:
    def __init__(self, max_queue_depth: int, timeout_s: float) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._max_queue_depth = max_queue_depth
        self._timeout_s = timeout_s
        self._waiting = 0

    @property
    def queue_depth(self) -> int:
        return self._waiting

    async def run(self, fn: Callable[..., T], *args: Any) -> tuple[T, float]:
        """Run `fn` on the inference worker. Returns (result, elapsed_ms)."""
        if self._waiting >= self._max_queue_depth:
            raise Busy(
                f"{self._waiting} requests already waiting (limit {self._max_queue_depth})"
            )
        self._waiting += 1
        started = time.perf_counter()
        try:
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(self._pool, fn, *args)
            # shield so that abandoning the wait does not try to cancel a thread
            # that cannot be cancelled. The work finishes and keeps occupying
            # the single worker, which is the backpressure we want.
            result = await asyncio.wait_for(asyncio.shield(fut), self._timeout_s)
        except asyncio.TimeoutError as exc:
            raise InferenceTimeout(
                f"inference exceeded {self._timeout_s}s — the model is stuck or the "
                f"machine is thermally throttled"
            ) from exc
        finally:
            self._waiting -= 1
        return result, round((time.perf_counter() - started) * 1000.0, 2)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
