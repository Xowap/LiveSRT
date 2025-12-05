import asyncio
import functools
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def sync_to_async(fn: Callable[P, R]) -> Callable[P, Awaitable[R]]:
    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        loop = asyncio.get_event_loop()
        p_func = functools.partial(fn, *args, **kwargs)
        return await loop.run_in_executor(None, p_func)

    return wrapper


def run_sync(fn: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, R]:
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return asyncio.run(fn(*args, **kwargs))

    return wrapper
