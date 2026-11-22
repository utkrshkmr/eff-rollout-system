"""CUPTI activity tracing with named phases.

Uses NVIDIA's `cupti-python` bindings (the CUPTI Activity API). Every kernel,
memcpy, memset, CUDA runtime call and CUDA driver call made by this process is
recorded by CUPTI and assigned to the phase that was open when it was launched.

Phase attribution works in two ways:
  1. External correlation: on `phase(name)` entry an id is pushed with
     `cuptiActivityPushExternalCorrelationId`; CUPTI then emits an
     EXTERNAL_CORRELATION record that links each runtime/driver API call made
     inside the phase to that id, and each kernel/memcpy carries the
     correlation id of the API call that launched it (for CUDA graphs: the
     cudaGraphLaunch call). This is exact even if kernels run asynchronously.
  2. Time window fallback: activity whose correlation id is unknown (e.g.
     launched from a thread other than the one that opened the phase) is
     assigned by GPU start time to the innermost phase window. Phase windows
     are bracketed by `torch.cuda.synchronize()` so the fallback is exact for
     this single-GPU, sequential pipeline.

Usage:
    prof = CuptiProfiler()
    prof.start()
    with prof.phase("rollout"):
        ...
    summary = prof.stop_and_summarize()      # dict, JSON-serializable
    prof.write_chrome_trace("trace.json.gz") # open in https://ui.perfetto.dev
"""
from __future__ import annotations

import gzip
import json
import re
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch

try:
    from cupti import cupti
    _CUPTI_IMPORT_ERROR: Optional[BaseException] = None
except Exception as e:  # pragma: no cover - reported by check_env
    cupti = None
    _CUPTI_IMPORT_ERROR = e

BUFFER_BYTES = 32 << 20

# Kernel name -> category. First match wins; order matters.
KERNEL_CATEGORIES: List[Tuple[str, re.Pattern]] = [
    ("nccl", re.compile(r"nccl", re.I)),
    ("attention", re.compile(r"flash|fmha|attn|attention|paged|mha_|_fwd_kernel|_bwd_kernel|reshape_and_cache", re.I)),
    # vLLM's quantized GEMMs: Marlin (weight-only FP8/INT4), Cutlass scaled_mm (its kernels are wrapped in
    # enable_sm80_to_sm89 / enable_sm90_or_later; int8 operands mangle as "...Visitor Ia" = signed char)
    ("gemm_quant", re.compile(r"marlin|scaled_mm|enable_sm\d+_(to_sm\d+|or_later)|GemmWithVisitorIa|w8a8|i16832gemm|"
                              r"gemm.*(int8|_s8_|e4m3)", re.I)),
    ("gemm", re.compile(r"gemm|cutlass|xmma|cublas|s16816|s1688|h16816|_mma_|splitk|Kernel2|nvjet|ampere_.*(tn|nt|nn|tt)|sm\d+_.*(tn|nt|nn|tt)", re.I)),
    ("quantize", re.compile(r"quant|fp8|int8|e4m3", re.I)),
    ("optimizer", re.compile(r"adam|multi_tensor|foreach|fused_.*optim|lamb|sgd", re.I)),
    ("norm", re.compile(r"rms|layer_?norm|norm_kernel|_norm", re.I)),
    ("softmax_logprob", re.compile(r"softmax|logsumexp|cross_entropy|nll|log_softmax", re.I)),
    ("sampling", re.compile(r"topk|top_k|top_p|sort|sampl|argmax|multinomial|radix|gumbel|penalt", re.I)),
    ("rotary", re.compile(r"rotary|rope", re.I)),
    ("inductor_triton", re.compile(r"^triton_|triton", re.I)),
    ("reduce", re.compile(r"reduce", re.I)),
    ("copy_cast", re.compile(r"copy|cast|cat_|catarray|index|gather|scatter|embedding|fill|memset|clone|transpose|permute", re.I)),
    ("elementwise", re.compile(r"elementwise|vectorized|unrolled|pointwise|silu|gelu|act_and_mul|mul|add", re.I)),
]


def categorize(name: str) -> str:
    for cat, pat in KERNEL_CATEGORIES:
        if pat.search(name):
            return cat
    return "other"


def cupti_available() -> Tuple[bool, str]:
    if cupti is None:
        return False, f"cupti-python import failed: {_CUPTI_IMPORT_ERROR!r}"
    from importlib.metadata import version
    api = cupti.get_version() if hasattr(cupti, "get_version") else "n/a"   # get_version: cupti-python >= 13.2
    return True, f"cupti-python {version('cupti-python')}, libcupti {version('nvidia-cuda-cupti')}, CUPTI API {api}"


class _Phase:
    __slots__ = ("name", "ext_id", "t0", "t1", "wall0", "wall1", "parent", "mem0", "mem_peak")

    def __init__(self, name, ext_id, parent):
        self.name, self.ext_id, self.parent = name, ext_id, parent
        self.t0 = self.t1 = 0
        self.wall0 = self.wall1 = 0.0
        self.mem0 = self.mem_peak = 0


class CuptiProfiler:
    """Collects CUPTI activity records and attributes them to named phases.

    `enabled=False` keeps the phase API (wall-clock timing, memory peaks) but
    records no CUPTI activity, so unprofiled steps cost nothing extra.
    """

    def __init__(self, record_api: bool = True, sync: bool = True):
        self.record_api = record_api
        self.sync = sync
        self.enabled = False
        self._lock = threading.Lock()
        self._kernels: List[tuple] = []   # (corr, start, end, name, stream, graph_id, regs, grid, block)
        self._copies: List[tuple] = []    # (corr, start, end, copy_kind, bytes, stream)
        self._memsets: List[tuple] = []   # (corr, start, end, bytes, stream)
        self._apis: List[tuple] = []      # (corr, start, end, domain, cbid, tid)
        self._ext: Dict[int, int] = {}    # correlation id -> external (phase) id
        self._phases: List[_Phase] = []
        self._stack: List[_Phase] = []
        self._next_ext = 1
        self._registered = False
        self._callback_names: Dict[Tuple[str, int], str] = {}
        self.dropped = 0

    # ---------------------------------------------------------------- CUPTI
    def _buffer_requested(self):
        return BUFFER_BYTES, 0

    def _buffer_completed(self, activities):
        K = cupti.ActivityKind
        kern, cps, mss, apis, ext = [], [], [], [], {}
        for a in activities:
            k = a.kind
            if k == K.CONCURRENT_KERNEL or k == K.KERNEL:
                kern.append((a.correlation_id, a.start, a.end, a.name, a.stream_id, a.graph_id,
                             a.registers_per_thread, a.grid_x * a.grid_y * a.grid_z,
                             a.block_x * a.block_y * a.block_z))
            elif k == K.MEMCPY:
                cps.append((a.correlation_id, a.start, a.end, int(a.copy_kind), a.bytes, a.stream_id))
            elif k == K.MEMSET:
                mss.append((a.correlation_id, a.start, a.end, a.bytes, a.stream_id))
            elif k == K.RUNTIME:
                apis.append((a.correlation_id, a.start, a.end, "runtime", a.cbid, a.thread_id))
            elif k == K.DRIVER:
                apis.append((a.correlation_id, a.start, a.end, "driver", a.cbid, a.thread_id))
            elif k == K.EXTERNAL_CORRELATION:
                ext[a.correlation_id] = a.external_id
        with self._lock:
            self._kernels += kern
            self._copies += cps
            self._memsets += mss
            self._apis += apis
            self._ext.update(ext)

    def _kinds(self):
        K = cupti.ActivityKind
        kinds = [K.CONCURRENT_KERNEL, K.MEMCPY, K.MEMSET, K.EXTERNAL_CORRELATION]
        if self.record_api:
            kinds += [K.RUNTIME, K.DRIVER]
        return kinds

    def start(self):
        """Begin recording CUPTI activity (call once per profiled window)."""
        ok, msg = cupti_available()
        if not ok:
            raise RuntimeError(msg)
        torch.cuda.synchronize()
        if not self._registered:
            cupti.activity_register_callbacks(self._buffer_requested, self._buffer_completed)
            self._registered = True
        for k in self._kinds():
            cupti.activity_enable(k)
        self.enabled = True

    def stop(self):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        cupti.activity_flush_all(1)
        for k in self._kinds():
            cupti.activity_disable(k)
        cupti.activity_flush_all(1)
        self.enabled = False

    def reset(self):
        with self._lock:
            self._kernels, self._copies, self._memsets, self._apis = [], [], [], []
            self._ext = {}
        self._phases, self._stack = [], []

    # --------------------------------------------------------------- phases
    def _now(self) -> int:
        return cupti.get_timestamp() if cupti is not None else time.perf_counter_ns()

    @contextmanager
    def phase(self, name: str):
        """Named region. Nested phases are joined with '/', e.g. 'train/backward'."""
        parent = self._stack[-1] if self._stack else None
        full = f"{parent.name}/{name}" if parent else name
        ph = _Phase(full, self._next_ext, parent)
        self._next_ext += 1
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.cuda.is_available():
            ph.mem0 = torch.cuda.memory_allocated()
            if parent is not None:  # save the parent's peak so far before resetting the counter
                parent.mem_peak = max(parent.mem_peak, torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
        ph.wall0 = time.perf_counter()
        ph.t0 = self._now()
        if self.enabled:
            cupti.activity_push_external_correlation_id(cupti.ExternalCorrelationKind.CUSTOM0, ph.ext_id)
        self._stack.append(ph)
        try:
            yield ph
        finally:
            if self.sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            ph.t1 = self._now()
            ph.wall1 = time.perf_counter()
            if torch.cuda.is_available():
                ph.mem_peak = max(ph.mem_peak, torch.cuda.max_memory_allocated())
                if parent is not None:  # keep the parent's peak monotone
                    parent.mem_peak = max(parent.mem_peak, ph.mem_peak)
            if self.enabled:
                cupti.activity_pop_external_correlation_id(cupti.ExternalCorrelationKind.CUSTOM0)
            self._stack.pop()
            self._phases.append(ph)

    def wall_times(self) -> Dict[str, float]:
        out: Dict[str, float] = defaultdict(float)
        for p in self._phases:
            out[p.name] += p.wall1 - p.wall0
        return dict(out)

    # ------------------------------------------------------------- analysis
    def _api_name(self, domain: str, cbid: int) -> str:
        key = (domain, cbid)
        if key not in self._callback_names:
            try:
                dom = cupti.CallbackDomain.RUNTIME_API if domain == "runtime" else cupti.CallbackDomain.DRIVER_API
                self._callback_names[key] = cupti.get_callback_name(dom, cbid)
            except Exception:
                self._callback_names[key] = f"{domain}:{cbid}"
        return self._callback_names[key]

    def _assigner(self):
        by_ext = {p.ext_id: p for p in self._phases}
        # innermost window lookup: phases sorted by duration (shortest first)
        windows = sorted(self._phases, key=lambda p: p.t1 - p.t0)

        def assign(corr: int, start: int) -> Tuple[str, bool]:
            e = self._ext.get(corr)
            if e is not None and e in by_ext:
                return by_ext[e].name, True
            for p in windows:
                if p.t0 <= start <= p.t1:
                    return p.name, False
            return "(outside phases)", False
        return assign

    def summarize(self, top_k: int = 15) -> dict:
        """Per-phase GPU activity summary."""
        assign = self._assigner()
        phases: Dict[str, dict] = {}

        def P(name):
            if name not in phases:
                phases[name] = dict(kernels=0, kernel_ns=0, by_category=defaultdict(lambda: [0, 0]),
                                    by_kernel=defaultdict(lambda: [0, 0]), memcpy=defaultdict(lambda: [0, 0, 0]),
                                    memset_bytes=0, memset_count=0, api=defaultdict(lambda: [0, 0]),
                                    graph_kernels=0, intervals=[], attributed_exact=0, attributed_window=0)
            return phases[name]

        copy_names = {int(v): v.name for v in cupti.ActivityMemcpyKind} if cupti else {}
        for corr, s, e, name, stream, graph_id, regs, grid, block in self._kernels:
            ph, exact = assign(corr, s)
            d = P(ph)
            d["kernels"] += 1
            d["kernel_ns"] += e - s
            c = d["by_category"][categorize(name)]
            c[0] += 1; c[1] += e - s
            kk = d["by_kernel"][name]
            kk[0] += 1; kk[1] += e - s
            d["graph_kernels"] += 1 if graph_id else 0
            d["intervals"].append((s, e))
            d["attributed_exact" if exact else "attributed_window"] += 1
        for corr, s, e, kind, nbytes, stream in self._copies:
            ph, _ = assign(corr, s)
            m = P(ph)["memcpy"][copy_names.get(kind, str(kind))]
            m[0] += 1; m[1] += nbytes; m[2] += e - s
            P(ph)["intervals"].append((s, e))
        for corr, s, e, nbytes, stream in self._memsets:
            ph, _ = assign(corr, s)
            P(ph)["memset_bytes"] += nbytes
            P(ph)["memset_count"] += 1
        for corr, s, e, domain, cbid, tid in self._apis:
            ph, _ = assign(corr, s)
            a = P(ph)["api"][self._api_name(domain, cbid)]
            a[0] += 1; a[1] += e - s

        wall = self.wall_times()
        out = {}
        for name, d in phases.items():
            busy = _union_ns(d.pop("intervals"))
            w = wall.get(name, 0.0)
            cats = {k: dict(count=v[0], ms=v[1] / 1e6) for k, v in
                    sorted(d["by_category"].items(), key=lambda kv: -kv[1][1])}
            top = [dict(name=k, count=v[0], ms=v[1] / 1e6, category=categorize(k)) for k, v in
                   sorted(d["by_kernel"].items(), key=lambda kv: -kv[1][1])[:top_k]]
            mem = {k: dict(count=v[0], MB=v[1] / 1e6, ms=v[2] / 1e6,
                           GBps=(v[1] / v[2]) if v[2] else 0.0) for k, v in d["memcpy"].items()}
            api = sorted(([k, v[0], v[1] / 1e6] for k, v in d["api"].items()), key=lambda x: -x[2])
            out[name] = dict(
                wall_ms=w * 1e3,
                gpu_busy_ms=busy / 1e6,
                gpu_busy_frac=(busy / 1e9 / w) if w else None,
                kernels=d["kernels"], kernel_ms=d["kernel_ns"] / 1e6,
                graph_kernels=d["graph_kernels"],
                kernels_attributed_by_correlation=d["attributed_exact"],
                kernels_attributed_by_time=d["attributed_window"],
                by_category=cats, top_kernels=top, memcpy=mem,
                memset=dict(count=d["memset_count"], MB=d["memset_bytes"] / 1e6),
                api_calls=sum(a[1] for a in api), api_ms=sum(a[2] for a in api),
                top_api=[dict(name=a[0], count=a[1], ms=a[2]) for a in api[:10]],
            )
        # phases that launched nothing still get wall time and memory
        for p in self._phases:
            o = out.setdefault(p.name, dict(wall_ms=wall.get(p.name, 0) * 1e3, kernels=0, kernel_ms=0.0,
                                            gpu_busy_ms=0.0, gpu_busy_frac=0.0))
            o["mem_peak_GB"] = max(o.get("mem_peak_GB", 0.0), p.mem_peak / 1e9)
            o["mem_start_GB"] = p.mem0 / 1e9
        return dict(phases=out, total_kernels=len(self._kernels), total_copies=len(self._copies),
                    total_api_calls=len(self._apis))

    def write_chrome_trace(self, path: str, max_events: int = 2_000_000):
        """Chrome/Perfetto trace: one track per phase (host windows) plus GPU streams."""
        assign = self._assigner()
        ev = []
        base = min([p.t0 for p in self._phases] or [0])
        us = lambda t: (t - base) / 1e3
        for p in self._phases:
            ev.append(dict(name=p.name, ph="X", pid="phases", tid=p.name.split("/")[0],
                           ts=us(p.t0), dur=(p.t1 - p.t0) / 1e3))
        for corr, s, e, name, stream, graph_id, regs, grid, block in self._kernels[:max_events]:
            ph, _ = assign(corr, s)
            ev.append(dict(name=name[:200], ph="X", pid="gpu", tid=f"stream {stream}", ts=us(s),
                           dur=(e - s) / 1e3, cat=categorize(name),
                           args=dict(phase=ph, grid=grid, block=block, regs=regs, graph=graph_id)))
        copy_names = {int(v): v.name for v in cupti.ActivityMemcpyKind} if cupti else {}
        for corr, s, e, kind, nbytes, stream in self._copies[:max_events]:
            ph, _ = assign(corr, s)
            ev.append(dict(name=f"memcpy {copy_names.get(kind, kind)}", ph="X", pid="gpu",
                           tid=f"stream {stream}", ts=us(s), dur=(e - s) / 1e3,
                           args=dict(phase=ph, bytes=nbytes)))
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "wt") as f:
            json.dump(dict(traceEvents=ev, displayTimeUnit="ms"), f)


def _union_ns(iv: List[Tuple[int, int]]) -> int:
    if not iv:
        return 0
    iv.sort()
    total, cs, ce = 0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        elif e > ce:
            ce = e
    return total + ce - cs
