from __future__ import annotations

from collections.abc import Callable
from contextlib import AsyncExitStack
from functools import partial
from typing import Any

import attrs
from anyio import CancelScope, CapacityLimiter, current_time, to_thread

from .._exceptions import JobTimedOutError
from .._structures import Job
from ..abc import JobExecutor


@attrs.define(eq=False, kw_only=True)
class ThreadPoolJobExecutor(JobExecutor):
    """
    Executes functions in a thread pool.

    :param max_workers: the maximum number of worker threads to keep
    """

    max_workers: int = 40
    _limiter: CapacityLimiter = attrs.field(init=False)

    async def start(self, exit_stack: AsyncExitStack) -> None:
        self._limiter = CapacityLimiter(self.max_workers)

    async def run_job(self, func: Callable[..., Any], job: Job) -> Any:
        wrapped = partial(func, *job.args, **job.kwargs)
        if job.timeout is None:
            return await to_thread.run_sync(wrapped, limiter=self._limiter)

        # Do not abandon a running thread on cancellation. The callable may be in the
        # middle of committing its result, and abandoning it would let the job continue
        # modifying data after the scheduler had already recorded a timeout.
        with CancelScope(
            deadline=current_time() + job.timeout.total_seconds()
        ) as timeout_scope:
            retval = await to_thread.run_sync(wrapped, limiter=self._limiter)

        if timeout_scope.cancel_called:
            raise JobTimedOutError

        return retval
