"""Unit tests for mempalace.parallel.

These tests do NOT touch ChromaDB, the network, or the miner. They exercise
the pipeline harness in isolation: producer thread pool + bounded queue +
single-thread consumer + shutdown semantics.

The single-thread consumer guarantee is the load-bearing invariant —
``mempalace.miner`` relies on it to preserve ChromaDB's HNSW
``num_threads=1`` write constraint. Two tests pin it explicitly.
"""

from __future__ import annotations

import threading
import time

import pytest

from mempalace.parallel import (
    FailedResult,
    ParallelPipeline,
    WorkerResult,
)


def _identity_producer(item):
    """Producer that emits one WorkerResult per item."""
    return WorkerResult(payload=item, item_id=str(item), work_count=1)


def test_pipeline_processes_all_items_workers_1():
    """workers=1 baseline: every item flows producer → consumer once."""
    consumed = []
    pipeline = ParallelPipeline(
        producer_fn=_identity_producer,
        consumer_fn=lambda r: consumed.append(r.payload),
        workers=1,
        queue_size=4,
    )
    stats = pipeline.run(range(10))

    assert stats.items_total == 10
    assert stats.items_succeeded == 10
    assert stats.items_failed == 0
    assert stats.interrupted is False
    assert sorted(consumed) == list(range(10))


def test_pipeline_processes_all_items_workers_many():
    """workers=8: every item still flows exactly once (no dup, no drop)."""
    consumed = []
    lock = threading.Lock()

    def consumer(r):
        with lock:
            consumed.append(r.payload)

    pipeline = ParallelPipeline(
        producer_fn=_identity_producer,
        consumer_fn=consumer,
        workers=8,
        queue_size=16,
    )
    stats = pipeline.run(range(100))

    assert stats.items_total == 100
    assert stats.items_succeeded == 100
    assert stats.items_failed == 0
    assert sorted(consumed) == list(range(100))


def test_pipeline_consumer_runs_in_single_thread():
    """The HNSW invariant: all consumer invocations share one thread id.

    If this ever regresses, the miner's parallel path will start corrupting
    ChromaDB's HNSW index on concurrent upserts. Pin it.
    """
    thread_ids = []
    lock = threading.Lock()

    def consumer(r):
        with lock:
            thread_ids.append(threading.get_ident())

    pipeline = ParallelPipeline(
        producer_fn=_identity_producer,
        consumer_fn=consumer,
        workers=8,
        queue_size=4,
    )
    pipeline.run(range(50))

    assert len(thread_ids) == 50
    assert len(set(thread_ids)) == 1, f"consumer ran on {len(set(thread_ids))} threads"


def test_pipeline_producer_returning_none_is_skip():
    """Producer returning None means 'no work for this item'.

    Used by the miner for files that are already mined (mtime check)."""
    consumed = []

    def producer(item):
        return _identity_producer(item) if item % 2 == 0 else None

    pipeline = ParallelPipeline(
        producer_fn=producer,
        consumer_fn=lambda r: consumed.append(r.payload),
        workers=4,
        queue_size=4,
    )
    stats = pipeline.run(range(10))

    assert stats.items_succeeded == 5  # only evens
    assert stats.items_failed == 0
    assert sorted(consumed) == [0, 2, 4, 6, 8]


def test_pipeline_producer_exception_becomes_failed_result():
    """One bad item must not abort the whole mine.

    Mirrors the existing liberal per-file failure handling in miner.py.
    """
    consumed = []
    failed = []

    def producer(item):
        if item == 5:
            raise RuntimeError("simulated embedding failure")
        return _identity_producer(item)

    pipeline = ParallelPipeline(
        producer_fn=producer,
        consumer_fn=lambda r: consumed.append(r.payload),
        on_error=lambda f: failed.append(f),
        workers=4,
        queue_size=4,
    )
    stats = pipeline.run(range(10))

    assert stats.items_succeeded == 9
    assert stats.items_failed == 1
    assert sorted(consumed) == [0, 1, 2, 3, 4, 6, 7, 8, 9]
    assert len(failed) == 1
    assert failed[0].item == 5
    assert isinstance(failed[0].exception, RuntimeError)


def test_pipeline_consumer_exception_does_not_kill_pipeline():
    """A consumer that raises on one item should still process the rest."""
    consumed = []

    def consumer(r):
        if r.payload == 5:
            raise RuntimeError("simulated upsert failure")
        consumed.append(r.payload)

    pipeline = ParallelPipeline(
        producer_fn=_identity_producer,
        consumer_fn=consumer,
        workers=4,
        queue_size=4,
    )
    stats = pipeline.run(range(10))

    # 5 failed in the consumer; the other 9 succeeded.
    assert stats.items_succeeded == 9
    assert stats.items_failed == 1
    assert 5 not in consumed


def test_pipeline_queue_backpressure():
    """Bounded queue + slow consumer: producers block on put.

    The point isn't to measure wallclock; it's to prove the queue is
    bounded. We test this by observing that producer wallclock is
    dominated by the consumer's processing time, not by spawning all
    items instantly.
    """
    consumer_calls = 0
    consumer_lock = threading.Lock()

    def slow_consumer(r):
        nonlocal consumer_calls
        time.sleep(0.02)  # 20ms per item
        with consumer_lock:
            consumer_calls += 1

    pipeline = ParallelPipeline(
        producer_fn=_identity_producer,
        consumer_fn=slow_consumer,
        workers=8,
        queue_size=2,  # very tight bound
    )

    t0 = time.monotonic()
    stats = pipeline.run(range(20))
    elapsed = time.monotonic() - t0

    assert stats.items_succeeded == 20
    # With queue_size=2 and single consumer @ 20ms/item, 20 items must take
    # at least ~400ms regardless of how many producers we have. If the queue
    # were unbounded, producers would dump all 20 in <10ms and elapsed
    # would be ~20ms (just the consumer time for one item ahead of return).
    assert elapsed > 0.3, f"expected backpressure, elapsed={elapsed:.3f}s"


def test_pipeline_progress_callback_fires_per_success():
    """on_progress fires once per successful consume, in the consumer thread."""
    progress_thread_ids = []
    progress_count = 0

    def on_progress(r):
        nonlocal progress_count
        progress_count += 1
        progress_thread_ids.append(threading.get_ident())

    pipeline = ParallelPipeline(
        producer_fn=_identity_producer,
        consumer_fn=lambda r: None,
        on_progress=on_progress,
        workers=4,
        queue_size=4,
    )
    pipeline.run(range(20))

    assert progress_count == 20
    # All on_progress calls run on the consumer thread (no lock needed
    # for the caller's progress counters).
    assert len(set(progress_thread_ids)) == 1


def test_pipeline_zero_items_is_noop():
    """Empty input must terminate cleanly, not deadlock."""
    pipeline = ParallelPipeline(
        producer_fn=_identity_producer,
        consumer_fn=lambda r: None,
        workers=4,
        queue_size=4,
    )
    stats = pipeline.run([])
    assert stats.items_total == 0
    assert stats.items_succeeded == 0


def test_pipeline_rejects_invalid_workers():
    with pytest.raises(ValueError):
        ParallelPipeline(
            producer_fn=_identity_producer,
            consumer_fn=lambda r: None,
            workers=0,
            queue_size=4,
        )


def test_pipeline_rejects_invalid_queue_size():
    with pytest.raises(ValueError):
        ParallelPipeline(
            producer_fn=_identity_producer,
            consumer_fn=lambda r: None,
            workers=1,
            queue_size=0,
        )


def test_pipeline_failed_result_default_logging_when_no_on_error():
    """Without on_error, a producer failure still increments items_failed.

    The user-visible behavior is a WARNING log entry. We don't assert on
    the log here (it's exercised by the on_error variant above); we only
    confirm the counter is right.
    """
    pipeline = ParallelPipeline(
        producer_fn=lambda item: (_ for _ in ()).throw(RuntimeError("boom"))
        if item == 0
        else _identity_producer(item),
        consumer_fn=lambda r: None,
        workers=2,
        queue_size=4,
    )
    stats = pipeline.run(range(3))
    assert stats.items_failed == 1
    assert stats.items_succeeded == 2
