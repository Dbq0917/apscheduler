from __future__ import annotations

from collections.abc import Callable
from contextlib import AsyncExitStack
from functools import partial
from typing import Any

import attrs
from anyio import CancelScope, CapacityLimiter, current_time, to_process

from .._exceptions import JobTimedOutError
from .._structures import Job
from ..abc import JobExecutor


@attrs.define(eq=False, kw_only=True)
class ProcessPoolJobExecutor(JobExecutor):
    """
    Executes functions in a process pool.

    :param max_workers: the maximum number of worker processes to keep
    """

    max_workers: int = 40
    _limiter: CapacityLimiter = attrs.field(init=False)

    async def start(self, exit_stack: AsyncExitStack) -> None:
        self._limiter = CapacityLimiter(self.max_workers)

    async def run_job(self, func: Callable[..., Any], job: Job) -> Any:
        wrapped = partial(func, *job.args, **job.kwargs)
        if job.timeout is None:
            return await to_process.run_sync(
                wrapped, cancellable=True, limiter=self._limiter
            )

        # Let the process finish an in-flight commit instead of killing it as soon as
        # the timeout expires. The timeout is reported once the callable returns.
        with CancelScope(
            deadline=current_time() + job.timeout.total_seconds()
        ) as timeout_scope:
            retval = await to_process.run_sync(wrapped, limiter=self._limiter)

        if timeout_scope.cancel_called:
            raise JobTimedOutError

        return retval
