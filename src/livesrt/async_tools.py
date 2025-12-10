"""
Small tools to smooth out working with asyncio
"""

import asyncio
import functools
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any


def sync_to_async[**P, R](fn: Callable[P, R]) -> Callable[P, Awaitable[R]]:
    """
    Runs a sync function in a parallel thread that you can await with regular
    asyncio semantics. Can be used as a decorator.
    """

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        loop = asyncio.get_event_loop()
        p_func = functools.partial(fn, *args, **kwargs)
        return await loop.run_in_executor(None, p_func)

    return wrapper


def run_sync[**P, R](fn: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, R]:
    """Runs an async function in an async loop. Can be used as a decorator."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return asyncio.run(fn(*args, **kwargs))

    return wrapper
