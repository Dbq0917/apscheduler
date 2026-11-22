from __future__ import annotations

from collections.abc import Callable
from inspect import isawaitable
from typing import Any

from anyio import CancelScope, current_time

from .._exceptions import JobTimedOutError
from .._structures import Job
from ..abc import JobExecutor


class AsyncJobExecutor(JobExecutor):
    """
    Executes functions directly on the event loop thread.

    If the function returns a coroutine object (or another kind of awaitable), that is
    awaited on and its return value is used as the job's return value.
    """

    async def run_job(self, func: Callable[..., Any], job: Job) -> Any:
        if job.timeout is None:
            retval = func(*job.args, **job.kwargs)
            if isawaitable(retval):
                retval = await retval

            return retval

        with CancelScope(
            deadline=current_time() + job.timeout.total_seconds()
        ) as timeout_scope:
            retval = func(*job.args, **job.kwargs)
            if isawaitable(retval):
                retval = await retval

        if timeout_scope.cancel_called:
            raise JobTimedOutError

        return retval
