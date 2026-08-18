"""
OjasX Asynchronous Multi-Queue Command Streams & Event Synchronization.
Provides full cudaStream_t and cudaEvent_t parity over OpenCL Command Queues.

Usage:
    import torchcl

    s1 = torchcl.Stream()
    s2 = torchcl.Stream()

    with torchcl.stream(s1):
        a = torchcl.to_opencl(x)
        b = torchcl.matmul(a, w)

    with torchcl.stream(s2):
        c = torchcl.to_opencl(y)
        d = torchcl.matmul(c, v)

    torchcl.synchronize()
"""

from __future__ import annotations

import threading
import time
from typing import Optional
import pyopencl as cl

from torchcl.runtime.context import get_context, get_device, get_queue, _set_current_queue, _get_default_queue


class Event:
    """Represents a synchronization event in the OpenCL command stream."""

    def __init__(self, enable_timing: bool = True) -> None:
        self.enable_timing = enable_timing
        self.cl_event: Optional[cl.Event] = None

    def record(self, stream: Optional[Stream] = None) -> None:
        """Record the event in the specified (or current) stream."""
        queue = stream.queue if stream is not None else get_queue()
        self.cl_event = cl.enqueue_marker(queue)

    def wait(self, stream: Optional[Stream] = None) -> None:
        """Make the specified stream wait for this event to complete."""
        if self.cl_event is None:
            return
        queue = stream.queue if stream is not None else get_queue()
        cl.enqueue_barrier(queue, wait_for=[self.cl_event])

    def query(self) -> bool:
        """Return True if the event has completed execution."""
        if self.cl_event is None:
            return True
        status = self.cl_event.get_info(cl.event_info.COMMAND_EXECUTION_STATUS)
        return status == cl.command_execution_status.COMPLETE

    def synchronize(self) -> None:
        """Block host thread until this event has completed."""
        if self.cl_event is not None:
            self.cl_event.wait()

    def elapsed_time(self, end_event: Event) -> float:
        """Return elapsed time in milliseconds between this event and end_event."""
        if self.cl_event is None or end_event.cl_event is None:
            return 0.0
        self.synchronize()
        end_event.synchronize()
        start_ns = self.cl_event.get_profiling_info(cl.profiling_info.START)
        end_ns = end_event.cl_event.get_profiling_info(cl.profiling_info.END)
        return (end_ns - start_ns) / 1e6


class Stream:
    """An asynchronous sequence of OpenCL commands that execute in order."""

    def __init__(self, priority: int = 0) -> None:
        self.ctx = get_context()
        self.dev = get_device()
        self.queue = cl.CommandQueue(
            self.ctx,
            self.dev,
            properties=cl.command_queue_properties.PROFILING_ENABLE,
        )

    def synchronize(self) -> None:
        """Block until all commands on this stream have completed."""
        self.queue.finish()

    def wait_event(self, event: Event) -> None:
        """Make this stream wait for an event."""
        event.wait(self)

    def wait_stream(self, other_stream: Stream) -> None:
        """Make this stream wait for all currently queued work in other_stream."""
        event = Event()
        event.record(other_stream)
        self.wait_event(event)


class stream:
    """Context manager that changes the active command stream."""

    def __init__(self, stream_obj: Stream) -> None:
        self.stream_obj = stream_obj
        self.prev_queue: Optional[cl.CommandQueue] = None

    def __enter__(self) -> Stream:
        self.prev_queue = get_queue()
        _set_current_queue(self.stream_obj.queue)
        return self.stream_obj

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _set_current_queue(self.prev_queue)


def current_stream() -> Stream:
    """Return the currently active stream."""
    s = Stream.__new__(Stream)
    s.ctx = get_context()
    s.dev = get_device()
    s.queue = get_queue()
    return s


def default_stream() -> Stream:
    """Return the default root stream."""
    s = Stream.__new__(Stream)
    s.ctx = get_context()
    s.dev = get_device()
    s.queue = _get_default_queue()
    return s
