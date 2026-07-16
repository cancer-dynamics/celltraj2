"""Retrying, cross-process access coordination for celltraj2 HDF5 files.

HDF5 already protects files with whole-file locks.  This module adds a small
cooperative lock in front of HDF5 so SITE processes can wait instead of failing
immediately, report useful wait events, and keep read/write lifetimes explicit.
The HDF5 lock remains enabled and is still the final authority.
"""

from __future__ import annotations

import errno
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


DEFAULT_READ_TIMEOUT = 60.0
DEFAULT_WRITE_TIMEOUT = 600.0
WAIT_EVENT_INTERVAL = 5.0

AccessReporter = Callable[[Mapping[str, Any]], None]


class H5AccessTimeout(TimeoutError):
    """Raised when a cooperative or native HDF5 lock stays busy too long."""


class H5DependencyChangedError(RuntimeError):
    """Raised when an input resource changed while a calculation was running."""


def _timeout_from_environment(mode: str, value: float | None) -> float:
    if value is not None:
        return max(0.0, float(value))
    key = "CELLTRAJ2_H5_READ_TIMEOUT" if mode == "r" else "CELLTRAJ2_H5_WRITE_TIMEOUT"
    fallback = DEFAULT_READ_TIMEOUT if mode == "r" else DEFAULT_WRITE_TIMEOUT
    try:
        return max(0.0, float(os.environ.get(key, fallback)))
    except (TypeError, ValueError):
        return fallback


def lock_path(path: str | Path) -> Path:
    """Return the cooperative sidecar lock path for an H5 file."""

    target = Path(path)
    return target.with_name(f".{target.name}.sitelab.lock")


def _emit(reporter: AccessReporter | None, event: Mapping[str, Any]) -> None:
    if reporter is not None:
        reporter(dict(event))


def _retry_delay(attempt: int) -> float:
    base = min(2.0, 0.1 * (1.6 ** max(0, attempt - 1)))
    return base * random.uniform(0.8, 1.2)


def _ensure_lock_byte(handle: Any) -> None:
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)


def _try_lock_windows(handle: Any, *, exclusive: bool) -> tuple[bool, Any]:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class OVERLAPPED(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_void_p),
            ("InternalHigh", ctypes.c_void_p),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    lock_file_ex = ctypes.windll.kernel32.LockFileEx
    lock_file_ex.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(OVERLAPPED),
    ]
    lock_file_ex.restype = wintypes.BOOL
    overlapped = OVERLAPPED()
    flags = 0x00000001  # LOCKFILE_FAIL_IMMEDIATELY
    if exclusive:
        flags |= 0x00000002  # LOCKFILE_EXCLUSIVE_LOCK
    os_handle = wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno()))
    ok = bool(lock_file_ex(os_handle, flags, 0, 1, 0, ctypes.byref(overlapped)))
    return ok, overlapped


def _unlock_windows(handle: Any, token: Any) -> None:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    unlock_file_ex = ctypes.windll.kernel32.UnlockFileEx
    unlock_file_ex.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    unlock_file_ex.restype = wintypes.BOOL
    os_handle = wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno()))
    unlock_file_ex(os_handle, 0, 1, 0, ctypes.byref(token))


def _try_lock(handle: Any, *, exclusive: bool) -> tuple[bool, Any]:
    if os.name == "nt":
        return _try_lock_windows(handle, exclusive=exclusive)
    import fcntl

    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        fcntl.flock(handle.fileno(), operation | fcntl.LOCK_NB)
    except BlockingIOError:
        return False, None
    return True, None


def _unlock(handle: Any, token: Any) -> None:
    if os.name == "nt":
        _unlock_windows(handle, token)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def file_lease(
    path: str | Path,
    *,
    exclusive: bool,
    timeout: float,
    reporter: AccessReporter | None = None,
    operation: str | None = None,
    job_id: str | None = None,
) -> Iterator[None]:
    """Hold a cooperative shared-reader or exclusive-writer lease."""

    target = Path(path)
    sidecar = lock_path(target)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + max(0.0, float(timeout))
    next_report = started
    attempt = 0
    handle = sidecar.open("a+b")
    token: Any = None
    try:
        _ensure_lock_byte(handle)
        while True:
            attempt += 1
            acquired, token = _try_lock(handle, exclusive=exclusive)
            now = time.monotonic()
            if acquired:
                waited = now - started
                if waited >= 0.05:
                    _emit(
                        reporter,
                        {
                            "event": "h5_lock_acquired",
                            "h5_path": str(target),
                            "mode": "write" if exclusive else "read",
                            "operation": operation,
                            "job_id": job_id,
                            "wait_seconds": waited,
                            "attempt": attempt,
                        },
                    )
                yield
                return
            if now >= deadline:
                raise H5AccessTimeout(
                    f"Timed out after {now - started:.1f}s waiting for "
                    f"{'write' if exclusive else 'read'} access to {target}"
                )
            if now >= next_report:
                _emit(
                    reporter,
                    {
                        "event": "h5_lock_waiting",
                        "h5_path": str(target),
                        "mode": "write" if exclusive else "read",
                        "operation": operation,
                        "job_id": job_id,
                        "wait_seconds": now - started,
                        "attempt": attempt,
                    },
                )
                next_report = now + WAIT_EVENT_INTERVAL
            time.sleep(min(_retry_delay(attempt), max(0.0, deadline - now)))
    finally:
        if token is not None:
            try:
                _unlock(handle, token)
            except Exception:
                pass
        handle.close()


def _is_retryable_open_error(exc: BaseException) -> bool:
    text = str(exc).casefold()
    retry_text = (
        "unable to lock file",
        "resource temporarily unavailable",
        "already open for write",
        "sharing violation",
        "file is already open",
        "errno = 11",
        "errno = 16",
        "winerror 32",
        "winerror 33",
    )
    if any(value in text for value in retry_text):
        return True
    return isinstance(exc, OSError) and getattr(exc, "errno", None) in {errno.EAGAIN, errno.EBUSY}


@contextmanager
def open_h5(
    path: str | Path,
    mode: str = "r",
    *,
    timeout: float | None = None,
    reporter: AccessReporter | None = None,
    operation: str | None = None,
    job_id: str | None = None,
    **kwargs: Any,
) -> Iterator[Any]:
    """Open an HDF5 file with cooperative locking and bounded retries."""

    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("HDF5 access requires h5py") from exc

    access_mode = "r" if mode == "r" else "r+"
    wait_timeout = _timeout_from_environment(access_mode, timeout)
    target = Path(path)
    exclusive = mode != "r"
    with file_lease(
        target,
        exclusive=exclusive,
        timeout=wait_timeout,
        reporter=reporter,
        operation=operation,
        job_id=job_id,
    ):
        # Give a native HDF5 lock conflict its own full retry window.  Time
        # already spent waiting for cooperative SITE readers/writers should
        # not consume this second, independent timeout.
        started = time.monotonic()
        deadline = started + wait_timeout
        attempt = 0
        next_report = started
        while True:
            attempt += 1
            try:
                handle = h5py.File(target, mode, **kwargs)
            except Exception as exc:
                if not _is_retryable_open_error(exc):
                    raise
                now = time.monotonic()
                if now >= deadline:
                    raise H5AccessTimeout(
                        f"Timed out after {now - started:.1f}s opening {target} in {mode!r} mode"
                    ) from exc
                if now >= next_report:
                    _emit(
                        reporter,
                        {
                            "event": "h5_native_lock_waiting",
                            "h5_path": str(target),
                            "mode": access_mode,
                            "operation": operation,
                            "job_id": job_id,
                            "wait_seconds": now - started,
                            "attempt": attempt,
                            "error": str(exc),
                        },
                    )
                    next_report = now + WAIT_EVENT_INTERVAL
                time.sleep(min(_retry_delay(attempt), max(0.0, deadline - now)))
                continue
            try:
                yield handle
            finally:
                if mode != "r":
                    handle.flush()
                handle.close()
            return


def snapshot_revisions(store: Any, paths: list[str] | tuple[str, ...]) -> dict[str, int]:
    """Capture resource revisions for later optimistic commit validation."""

    return {str(path): int(store.resource_revision(path)) for path in paths}


def validate_revisions(store: Any, expected: Mapping[str, int]) -> None:
    """Raise if a declared input resource changed during calculation."""

    changed = {
        str(path): (int(revision), int(store.resource_revision(path)))
        for path, revision in expected.items()
        if int(store.resource_revision(path)) != int(revision)
    }
    if changed:
        detail = ", ".join(
            f"{path}: expected {old}, found {new}" for path, (old, new) in changed.items()
        )
        raise H5DependencyChangedError(f"H5 dependencies changed during calculation: {detail}")


def run_with_stale_retries(
    action: Callable[[], Any],
    *,
    reporter: AccessReporter | None = None,
    context: Mapping[str, Any] | None = None,
    attempts: int = 3,
) -> Any:
    """Recalculate when optimistic dependency validation rejects a commit."""

    total = max(1, int(attempts))
    for attempt in range(1, total + 1):
        try:
            return action()
        except H5DependencyChangedError as exc:
            _emit(
                reporter,
                {
                    **dict(context or {}),
                    "event": "commit_stale",
                    "attempt": attempt,
                    "max_attempts": total,
                    "recomputing": attempt < total,
                    "error": str(exc),
                },
            )
            if attempt >= total:
                raise


__all__ = [
    "DEFAULT_READ_TIMEOUT",
    "DEFAULT_WRITE_TIMEOUT",
    "H5AccessTimeout",
    "H5DependencyChangedError",
    "file_lease",
    "lock_path",
    "open_h5",
    "run_with_stale_retries",
    "snapshot_revisions",
    "validate_revisions",
]
