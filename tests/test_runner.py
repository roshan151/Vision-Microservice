"""The serialization and backpressure guarantees, tested directly.

Uses asyncio.run rather than pytest-asyncio so the suite needs no extra plugin.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from vision_service.runner import Busy, InferenceRunner, InferenceTimeout


def test_inference_is_serialized():
    """One GPU, one worker: no two calls may overlap."""
    runner = InferenceRunner(max_queue_depth=8, timeout_s=5.0)
    state = {"concurrent": 0, "peak": 0}
    lock = threading.Lock()

    def work(_):
        with lock:
            state["concurrent"] += 1
            state["peak"] = max(state["peak"], state["concurrent"])
        time.sleep(0.02)
        with lock:
            state["concurrent"] -= 1
        return "done"

    async def main():
        return await asyncio.gather(*(runner.run(work, i) for i in range(6)))

    asyncio.run(main())
    assert state["peak"] == 1, f"inference overlapped: peak concurrency {state['peak']}"
    runner.shutdown()


def test_refuses_rather_than_queues_when_saturated():
    runner = InferenceRunner(max_queue_depth=2, timeout_s=5.0)

    def slow(_):
        time.sleep(0.2)
        return True

    async def main():
        return await asyncio.gather(
            *(runner.run(slow, i) for i in range(5)), return_exceptions=True
        )

    results = asyncio.run(main())
    assert any(isinstance(r, Busy) for r in results), "saturation should refuse, not queue"
    runner.shutdown()


def test_timeout_surfaces():
    runner = InferenceRunner(max_queue_depth=4, timeout_s=0.05)

    async def main():
        await runner.run(lambda _: time.sleep(0.5), None)

    with pytest.raises(InferenceTimeout):
        asyncio.run(main())
    runner.shutdown()


def test_queue_depth_returns_to_zero():
    runner = InferenceRunner(max_queue_depth=4, timeout_s=5.0)
    asyncio.run(runner.run(lambda _: 1, None))
    assert runner.queue_depth == 0
    runner.shutdown()


def test_elapsed_ms_is_reported():
    runner = InferenceRunner(max_queue_depth=4, timeout_s=5.0)
    _, ms = asyncio.run(runner.run(lambda _: time.sleep(0.03), None))
    assert ms >= 25.0
    runner.shutdown()
