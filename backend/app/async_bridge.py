"""
Utilities for safely running async callables from synchronous code.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


def _run_coroutine(async_fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
    """Run an async callable in a fresh event loop."""
    return asyncio.run(async_fn(*args, **kwargs))


def run_async_from_sync(
    async_fn: Callable[..., Awaitable[T]],
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Execute an async callable from sync code.

    If no loop is running in the current thread, run directly with ``asyncio.run``.
    If a loop is already running, execute in a dedicated worker thread with its own loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_coroutine(async_fn, *args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_coroutine, async_fn, *args, **kwargs)
        return future.result()
