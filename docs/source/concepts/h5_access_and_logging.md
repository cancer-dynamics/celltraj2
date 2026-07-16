# H5 Access And Job Logging

`celltraj2` coordinates every H5 open through `celltraj2.h5_access.open_h5`.
The goal is to keep the canonical `.ct2.h5` available to SITE readers during
long analyses while serializing only the mutations that must be exclusive.

## Access Model

Each H5 has a hidden cooperative sidecar named
`.name.ct2.h5.sitelab.lock`. Readers take a shared lease and may run
concurrently. A writer takes an exclusive lease, opens the canonical file with
`r+`, flushes and closes it before releasing the lease. Native HDF5 file
locking remains enabled and is the final authority; the sidecar does not
replace it.

Batch workers follow this sequence:

1. Open the canonical H5 read-only and calculate against a stable input view.
2. Close that read handle before requesting write access.
3. Wait for an exclusive lease, then open the canonical file with `r+` only
   for the result commit.
4. Revalidate the revisions of every declared input resource.
5. Write the result and compact final run provenance, flush, and close.

This means the ROI viewer, SITE Command Center, and other analysis processes
can share read access during calculation. If several jobs target one H5, their
read phases may overlap and their commits serialize. A completed reader can
therefore wait for another job's read phase before committing; it never keeps
an `r+` handle open while it waits or calculates.

The default wait is 60 seconds for a read and 600 seconds for a write. Set
`CELLTRAJ2_H5_READ_TIMEOUT` or `CELLTRAJ2_H5_WRITE_TIMEOUT` to change those
limits. Lock conflicts use jittered backoff and emit periodic wait events.
Native HDF5 lock conflicts are retried as well, which covers non-SITE readers
that do not use the cooperative sidecar.

## Concurrent Changes

Stored resources have monotonic revision attributes. A worker snapshots the
revisions of its scientific inputs before calculation and checks them again
inside the exclusive commit. If an input changed, the result is not written;
the worker reports `commit_stale`, reopens the current inputs, and recalculates
up to three times. Output existence is checked again at commit, so two jobs
with `overwrite=false` cannot silently replace one another.

This is optimistic concurrency, not snapshot isolation. It deliberately
allows multiple jobs to touch the same H5 rather than rejecting them at queue
submission. Native HDF5 locking still prevents simultaneous readers and a
writer. A process killed during an HDF5 mutation can never be made completely
risk-free without a copy-and-swap design, but short, flushed commits greatly
reduce that interruption window.

`Trajectory(path)` is read-only by default. Direct scripts that mutate a file
must opt in with `Trajectory(path, mode="r+")` and should keep that context as
short as possible. SITE and batch code should use the centralized access
helper rather than calling `h5py.File` directly.

## Logs And Progress

Per-frame and per-step progress is operational data, not scientific H5 data.
`JsonlReporter` writes every event immediately to unbuffered stdout and, when
`CELLTRAJ2_EVENTS_PATH` is set, appends and flushes the same event to a durable
JSONL file. Workers emit progress while the calculation is happening, rather
than collecting frame output until process completion.

Batch jobs keep only compact final provenance under `/runs/.../run.json`.
Detailed frame starts, summaries, skip/failure state, lock waits, stale-commit
retries, and backend output belong in the external job log and `events.jsonl`.
This removes repeated progress-only mutations from the H5 and leaves the H5
focused on canonical analysis results.
