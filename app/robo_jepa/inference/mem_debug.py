# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Lightweight host-RSS / CUDA-mem instrumentation for the bf16 OOM hunt.

All logging is gated on the ``RJP_RSS_DEBUG`` env var so production runs are
unaffected.  Set ``RJP_RSS_DEBUG=1`` to enable, ``RJP_RSS_DEBUG=2`` to also
log per-call deltas.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

_PSUTIL_PROC = None
_PREV_RSS_GB: Optional[float] = None
_T0 = time.perf_counter()


def _enabled() -> bool:
    return os.environ.get("RJP_RSS_DEBUG", "0") not in ("0", "", "false", "False")


def _verbose() -> bool:
    return os.environ.get("RJP_RSS_DEBUG", "0") in ("2", "verbose")


def _proc():
    global _PSUTIL_PROC
    if _PSUTIL_PROC is None:
        try:
            import psutil

            _PSUTIL_PROC = psutil.Process()
        except Exception:
            _PSUTIL_PROC = False
    return _PSUTIL_PROC


def log_rss(tag: str, *, force: bool = False) -> None:
    """Log host RSS + CUDA alloc/reserved tagged with ``tag``.

    No-op unless ``RJP_RSS_DEBUG=1`` (or ``force=True``).
    """
    if not (force or _enabled()):
        return
    proc = _proc()
    if not proc:
        return
    try:
        import torch

        rss_gb = proc.memory_info().rss / (1024**3)
        try:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            rank = 0

        cuda_alloc_gb = 0.0
        cuda_reserved_gb = 0.0
        cuda_max_alloc_gb = 0.0
        if torch.cuda.is_available():
            cuda_alloc_gb = torch.cuda.memory_allocated() / (1024**3)
            cuda_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            cuda_max_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3)

        global _PREV_RSS_GB
        delta = "" if _PREV_RSS_GB is None else f" Δrss={(rss_gb - _PREV_RSS_GB):+.2f}GB"
        _PREV_RSS_GB = rss_gb
        elapsed = time.perf_counter() - _T0
        logger.info(
            "[RSS rank=%d t=%.0fs] %-44s rss=%.2fGB%s cuda_alloc=%.2fGB "
            "cuda_reserved=%.2fGB cuda_max_alloc=%.2fGB",
            rank,
            elapsed,
            tag,
            rss_gb,
            delta,
            cuda_alloc_gb,
            cuda_reserved_gb,
            cuda_max_alloc_gb,
        )
    except Exception as exc:  # pragma: no cover - never let logging crash a run
        logger.debug("log_rss(%r) failed: %s", tag, exc)


def reset_cuda_peak() -> None:
    if not _enabled():
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
