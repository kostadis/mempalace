"""Producer/consumer pipeline for mining work.

Why this exists: the mining hot path is HTTP-bound on the embedding endpoint
(vLLM on a DGX Spark in the canonical setup). A serial mine leaves the GPU
idle most of the wallclock. But ChromaDB's HNSW index is pinned to
``num_threads=1`` (hnswlib's write path is not thread-safe), so we can't
just throw a thread pool at ``collection.upsert``.

The pattern this module implements:

* **N producer threads** do the IO-bound work (read file → chunk → call the
  embedding HTTP endpoint → emit a :class:`WorkerResult`).
* **One consumer thread** drains the queue and runs the single-writer side
  effect (``collection.upsert`` with pre-computed embeddings).

The single-writer guarantee is the **caller's** responsibility — this module
only promises that ``consumer_fn`` is invoked from exactly one thread.

Design constraints:

* Bounded queue → producers backpressure on a slow consumer; memory is
  capped regardless of corpus size.
* Producer exceptions become :class:`FailedResult` entries on the queue.
  The consumer logs / accounts them; the pipeline does not abort on a
  single bad file (mirrors the existing liberal per-file failure handling
  in ``mempalace.miner.process_file``).
* ``KeyboardInterrupt`` in the main thread sets a shutdown :class:`Event`;
  in-flight producers exit at the next ``queue.put`` checkpoint and the
  consumer drains what it already has. The main thread re-raises so the
  caller's existing resume-summary code path runs.
* No asyncio. Threading + ``queue.Queue`` is the right primitive for
  Python where the workload is HTTP-bound and the consumer side is a
  synchronous library call (chromadb).

This module is consumed by ``mempalace.miner._mine_impl`` and is designed
to be reused by ``convo_miner``, ``llm_refine``, and ``closet_llm`` once
the parallelism work extends to those code paths (see
``docs/design/embrace-parallelism.md``).
"""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class WorkerResult:
    """What a producer emits per item. Consumed by ``consumer_fn``.

    ``payload`` is consumer-defined — typically a struct holding everything
    needed to do the single-writer side effect (e.g., for the miner: drawer
    ids, documents, metadatas, pre-computed embeddings).
    """

    payload: Any
    item_id: str = ""
    """Short identifier for logging / progress callbacks (e.g., filename)."""
    work_count: int = 0
    """Unit count for the progress callback (e.g., drawer count)."""
    extra: dict = field(default_factory=dict)
    """Free-form bag for callback consumers (room name, etc.)."""


@dataclass
class FailedResult:
    """Emitted on the queue when a producer raises.

    The consumer receives this in place of a :class:`WorkerResult` and is
    expected to log + account it. The pipeline keeps running.
    """

    item: Any
    item_id: str
    exception: BaseException


@dataclass
class PipelineStats:
    """Returned from :meth:`ParallelPipeline.run`."""

    items_total: int = 0
    items_succeeded: int = 0
    items_failed: int = 0
    interrupted: bool = False


_SENTINEL = object()


class ParallelPipeline:
    """N producer threads + 1 consumer thread, bridged by a bounded queue.

    Usage::

        pipeline = ParallelPipeline(
            producer_fn=lambda item: WorkerResult(payload=embed(item), ...),
            consumer_fn=lambda result: write(result.payload),
            workers=8,
            queue_size=16,
        )
        stats = pipeline.run(items)

    The caller guarantees:

    * ``producer_fn`` is thread-safe (it will be called concurrently from N
      threads). Most ``producer_fn`` implementations only touch their own
      argument and a shared read-only reference (e.g., the embedding
      function callable), so this is usually trivial.
    * ``consumer_fn`` is the only path to any single-writer side effect.

    This class does **not** guarantee:

    * Ordering — items can be consumed in any order. If the caller needs
      ordering, it must do its own re-sequencing.
    * Cross-process safety — locks held by callbacks (e.g., the miner's
      ``mine_lock``) are honored by this module simply because the
      callbacks acquire them; this module adds no extra coordination.
    """

    def __init__(
        self,
        producer_fn: Callable[[Any], Optional[WorkerResult]],
        consumer_fn: Callable[[WorkerResult], None],
        *,
        workers: int,
        queue_size: int,
        on_progress: Optional[Callable[[WorkerResult], None]] = None,
        on_error: Optional[Callable[[FailedResult], None]] = None,
    ):
        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        if queue_size < 1:
            raise ValueError(f"queue_size must be >= 1, got {queue_size}")
        self.producer_fn = producer_fn
        self.consumer_fn = consumer_fn
        self.workers = workers
        self.queue_size = queue_size
        self.on_progress = on_progress
        self.on_error = on_error
        self._shutdown = threading.Event()
        self._consumer_thread_id: Optional[int] = None

    def run(self, items: Iterable[Any]) -> PipelineStats:
        """Pump ``items`` through the pipeline. Blocks until done.

        Returns :class:`PipelineStats` with success / failure counters. On
        ``KeyboardInterrupt`` in the main thread, sets the shutdown event,
        drains what the consumer has already received, then re-raises so
        the caller's existing summary-on-interrupt code path runs.
        """
        item_list = list(items)  # materialize once so we know the total
        stats = PipelineStats(items_total=len(item_list))

        q: queue.Queue = queue.Queue(maxsize=self.queue_size)
        consumer_thread = threading.Thread(
            target=self._consumer_loop,
            args=(q, stats),
            name="ParallelPipeline-consumer",
            daemon=True,
        )
        consumer_thread.start()

        try:
            with ThreadPoolExecutor(
                max_workers=self.workers,
                thread_name_prefix="ParallelPipeline-producer",
            ) as executor:
                futures: list[Future] = [
                    executor.submit(self._producer_loop, item, q) for item in item_list
                ]
                # Drain ALL producer futures — even after shutdown is set,
                # because the exception that triggered the shutdown is
                # stored in one of these futures and we need to surface it.
                # Post-shutdown futures are O(1) (the producer wrapper's
                # ``if _shutdown.is_set(): return`` early-exit), so draining
                # the tail is essentially free.
                #
                # KeyboardInterrupt from a producer (or from the main
                # thread waiting here) propagates up so the caller's
                # except KeyboardInterrupt handler runs — preserves the
                # serial-miner's resumable-mine-on-Ctrl-C contract.
                first_exc: Optional[BaseException] = None
                for fut in futures:
                    try:
                        fut.result()
                    except KeyboardInterrupt as exc:
                        # KI always wins — re-raise immediately so the
                        # caller sees it without waiting for later
                        # producers to drain.
                        self._shutdown.set()
                        raise exc
                    except Exception as exc:
                        # Non-KI producer-wrapper errors: log + remember
                        # the first one, but keep draining so all futures
                        # are resolved. Re-raise after the loop.
                        if first_exc is None:
                            logger.exception(
                                "producer wrapper raised; shutting down pipeline"
                            )
                            first_exc = exc
                            self._shutdown.set()
                if first_exc is not None:
                    raise first_exc
        except KeyboardInterrupt:
            stats.interrupted = True
            self._shutdown.set()
            # Don't re-raise yet — first let the consumer drain whatever it
            # has already taken off the queue. Then re-raise so the caller's
            # summary-on-interrupt code path runs.
            self._signal_consumer_done(q)
            consumer_thread.join(timeout=10.0)
            raise
        finally:
            if not stats.interrupted:
                self._signal_consumer_done(q)
                consumer_thread.join()

        return stats

    def _producer_loop(self, item: Any, q: queue.Queue) -> None:
        """Run one item through ``producer_fn`` and put result on the queue."""
        if self._shutdown.is_set():
            return
        try:
            result = self.producer_fn(item)
        except KeyboardInterrupt:
            # KI must propagate, not be silently converted to FailedResult.
            # Set shutdown so other producers exit at their next checkpoint
            # and re-raise into our caller (the executor), which will
            # surface it via Future.result() in the main thread.
            self._shutdown.set()
            raise
        except Exception as exc:  # noqa: BLE001 — must surface every error
            failed = FailedResult(
                item=item,
                item_id=str(item)[:80],
                exception=exc,
            )
            self._put_blocking(q, failed)
            return
        if result is None:
            # Producer chose to skip this item (e.g., file already mined).
            # Nothing to put on the queue.
            return
        self._put_blocking(q, result)

    def _put_blocking(self, q: queue.Queue, item: Any) -> None:
        """Put with periodic shutdown check so Ctrl-C doesn't deadlock."""
        while not self._shutdown.is_set():
            try:
                q.put(item, timeout=0.25)
                return
            except queue.Full:
                continue

    def _consumer_loop(self, q: queue.Queue, stats: PipelineStats) -> None:
        """Single-threaded drain loop."""
        self._consumer_thread_id = threading.get_ident()
        while True:
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                if self._shutdown.is_set():
                    # On shutdown, drain anything still on the queue
                    # without blocking, then exit.
                    self._drain_remaining(q, stats)
                    return
                continue

            if item is _SENTINEL:
                return

            if isinstance(item, FailedResult):
                stats.items_failed += 1
                if self.on_error is not None:
                    try:
                        self.on_error(item)
                    except BaseException:
                        logger.exception("on_error callback raised")
                else:
                    logger.warning(
                        "producer failed on %s: %s: %s",
                        item.item_id,
                        type(item.exception).__name__,
                        item.exception,
                    )
                continue

            # item is a WorkerResult
            try:
                self.consumer_fn(item)
            except BaseException:
                logger.exception("consumer_fn raised on %s", item.item_id)
                stats.items_failed += 1
                continue

            stats.items_succeeded += 1
            if self.on_progress is not None:
                try:
                    self.on_progress(item)
                except BaseException:
                    logger.exception("on_progress callback raised")

    def _drain_remaining(self, q: queue.Queue, stats: PipelineStats) -> None:
        """Best-effort drain of items already on the queue (no blocking)."""
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                return
            if item is _SENTINEL:
                return
            if isinstance(item, FailedResult):
                stats.items_failed += 1
                continue
            try:
                self.consumer_fn(item)
                stats.items_succeeded += 1
                if self.on_progress is not None:
                    try:
                        self.on_progress(item)
                    except BaseException:
                        logger.exception("on_progress callback raised during drain")
            except BaseException:
                logger.exception("consumer_fn raised during drain on %s", item.item_id)
                stats.items_failed += 1

    def _signal_consumer_done(self, q: queue.Queue) -> None:
        """Tell the consumer thread to exit after draining its current item."""
        try:
            q.put(_SENTINEL, timeout=5.0)
        except queue.Full:
            # Consumer is wedged. Force the issue.
            self._shutdown.set()


__all__ = [
    "FailedResult",
    "ParallelPipeline",
    "PipelineStats",
    "WorkerResult",
]
