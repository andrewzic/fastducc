# -*- coding: utf-8 -*-
"""
fastducc.gpu_nufft: Zero-copy GPU CSR bridge and Stage 2/3/4 NUFFT hand-off.

Connects direct GPU visibility rasterisation (:mod:`fastducc.gpu_direct_cube`
and :mod:`fastducc.ragged_gpu`) to:
  - :class:`periodicity.ragged_dga.GPURaggedDGAStore` and
    :class:`fastducc.ragged.RaggedDGAStore` via :func:`from_gpu_buffers`
    and :func:`gpu_store_to_host_ragged`.
  - ``cupyx.scipy.sparse.csr_matrix`` of shape ``(M_active, total_planes)``
    via :func:`build_gpu_csr_matrix`.
  - In-VRAM Stage 2 temporal NUFFT, Stage 3 batched 2D C2R synthesis, and
    Stage 4 harmonic peak detection via :func:`run_gpu_nufft_from_store`, with
    an explicit CPU fallback when CUDA is unavailable or ``device='cpu'``.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cupy as cp
    from cupy.cuda.runtime import CUDARuntimeError as _CUDARuntimeError

    HAS_CUPY = True
    _CUPY_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:
    cp = None  # type: ignore[assignment]
    _CUDARuntimeError = RuntimeError  # type: ignore[assignment,misc]
    HAS_CUPY = False
    _CUPY_IMPORT_ERROR = exc

try:
    import cupyx.scipy.sparse as cupy_sparse

    HAS_CUPY_SPARSE = True
except Exception:
    cupy_sparse = None  # type: ignore[assignment]
    HAS_CUPY_SPARSE = False

from fastducc.ragged import RaggedDGAStore as FastduccRaggedDGAStore

logger = logging.getLogger("fastducc.gpu_nufft")
_POWER_SYNC_BYTES = 512 * 1024 * 1024
_INT32_MAX = np.iinfo(np.int32).max


def _ensure_periodicity_importable() -> None:
    """Ensure sibling ``lotrun_periodicity/src`` is on ``sys.path`` if present."""
    try:
        import periodicity  # noqa: F401

        return
    except ImportError:
        pass

    fastducc_root = Path(__file__).resolve().parents[1]
    candidates = [
        os.environ.get("LOTRUN_PERIODICITY_DIR", ""),
        str(fastducc_root.parent / "lotrun_periodicity" / "src"),
        str(fastducc_root.parent / "lotrun_periodicity"),
    ]
    for cand in candidates:
        if not cand:
            continue
        src_path = os.path.join(cand, "src") if os.path.isdir(os.path.join(cand, "src", "periodicity")) else cand
        if os.path.isdir(os.path.join(src_path, "periodicity")) and src_path not in sys.path:
            sys.path.insert(0, src_path)
            break


def is_cuda_available() -> bool:
    """Return True if CuPy is importable and at least one CUDA device is accessible."""
    if not HAS_CUPY or cp is None:
        return False
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


def _require_cuda(device_id: int = 0) -> None:
    """Verify CuPy availability and activate the requested CUDA device."""
    if not HAS_CUPY or cp is None:
        raise RuntimeError(
            "CuPy is required for CUDA GPU NUFFT operations."
        ) from _CUPY_IMPORT_ERROR
    try:
        n_dev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError(f"CUDA device query failed: {exc}") from exc
    if n_dev <= 0:
        raise RuntimeError("CUDA execution requested, but no CUDA devices were detected.")
    cp.cuda.Device(int(device_id)).use()


def resolve_device(device: str = "auto") -> str:
    """
    Resolve compute backend selector ``device`` into ``'cuda'`` or ``'cpu'``.

    Parameters
    ----------
    device : {'auto', 'cuda', 'cpu'}
        Requested execution backend.
    """
    mode = str("auto" if device is None else device).strip().lower()
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        _require_cuda(0)
        return "cuda"
    if mode == "auto":
        if is_cuda_available():
            return "cuda"
        logger.debug("gpu_nufft(device='auto') falling back to CPU: CUDA/CuPy unavailable.")
        return "cpu"
    raise ValueError(f"Unsupported device='{device}'; expected 'auto', 'cuda', or 'cpu'.")


def _get_gpu_store_cls() -> Any:
    """Return :class:`periodicity.ragged_dga.GPURaggedDGAStore` or a compatible fallback."""
    _ensure_periodicity_importable()
    try:
        from periodicity.ragged_dga import GPURaggedDGAStore

        return GPURaggedDGAStore
    except ImportError:
        return _FallbackGPURaggedDGAStore


@dataclass
class _FallbackGPURaggedDGAStore:
    """Standalone GPU CSR container fallback used when ``periodicity`` is absent."""

    store_dir: str
    attrs: Dict[str, Any]
    cell_offsets: Any  # cp.ndarray (M + 1,) uint64
    cell_uv: Any       # cp.ndarray (M, 2) int32 [v, u]
    vis_data: Any      # cp.ndarray (NNZ,) complex64
    vis_times: Any     # cp.ndarray (NNZ,) float64
    device_id: int = 0

    @property
    def m_total(self) -> int:
        return int(self.cell_uv.shape[0]) if self.cell_uv is not None else 0

    @property
    def nnz(self) -> int:
        if self.cell_offsets is None or self.cell_offsets.size == 0:
            return 0
        with cp.cuda.Device(int(self.device_id)):
            return int(self.cell_offsets[-1].get())

    def to_cpu(self) -> FastduccRaggedDGAStore:
        """Copy device-resident arrays back to host NumPy arrays."""
        if self.cell_offsets is None or self.cell_uv is None or self.vis_data is None or self.vis_times is None:
            raise RuntimeError("GPURaggedDGAStore has already been released.")
        _require_cuda(self.device_id)
        with cp.cuda.Device(int(self.device_id)):
            return FastduccRaggedDGAStore(
                store_dir=self.store_dir,
                attrs=dict(self.attrs),
                cell_offsets=cp.asnumpy(self.cell_offsets),
                cell_uv=cp.asnumpy(self.cell_uv),
                vis_data=cp.asnumpy(self.vis_data),
                vis_times=cp.asnumpy(self.vis_times),
            )

    def to_csr_matrix(
        self,
        total_planes: Optional[int] = None,
        dt: Optional[float] = None,
        validate: bool = False,
    ) -> Any:
        """Build a zero-copy ``cupyx.scipy.sparse.csr_matrix`` view of this store."""
        eff_planes = int(self.attrs.get("total_planes", 0) if total_planes is None else total_planes)
        eff_dt = float(self.attrs.get("tsamp", 0.0) if dt is None else dt)
        return build_gpu_csr_matrix(self, total_planes=eff_planes, dt=eff_dt, validate=validate)

    def release(self) -> None:
        """Release device buffers and return unused blocks to the CuPy memory pool."""
        self.cell_offsets = None
        self.cell_uv = None
        self.vis_data = None
        self.vis_times = None
        if HAS_CUPY and cp is not None:
            with cp.cuda.Device(int(self.device_id)):
                cp.get_default_memory_pool().free_all_blocks()


def _infer_device_id(*arrays: Any) -> int:
    """Infer CUDA device ID from the first CuPy ndarray argument, defaulting to 0."""
    if HAS_CUPY and cp is not None:
        for arr in arrays:
            if isinstance(arr, cp.ndarray):
                return int(arr.device.id)
    return 0


def _as_gpu_zero_copy(arr: Any, dtype: Any, target_dev: int) -> "cp.ndarray":
    """Return a C-contiguous CuPy array of ``dtype`` on ``target_dev`` without unnecessary copies."""
    if isinstance(arr, cp.ndarray) and int(arr.device.id) == int(target_dev):
        if arr.dtype == cp.dtype(dtype) and arr.flags.c_contiguous:
            return arr
        return cp.ascontiguousarray(arr.astype(dtype, copy=False))
    if isinstance(arr, cp.ndarray):
        with cp.cuda.Device(int(arr.device.id)):
            host = cp.asnumpy(arr)
        with cp.cuda.Device(int(target_dev)):
            return cp.ascontiguousarray(cp.asarray(host, dtype=dtype))
    with cp.cuda.Device(int(target_dev)):
        return cp.ascontiguousarray(cp.asarray(arr, dtype=dtype))


def _as_host_contiguous(arr: Any, dtype: Any) -> np.ndarray:
    """Convert a NumPy or CuPy array into a C-contiguous host NumPy array of ``dtype``."""
    if HAS_CUPY and cp is not None and isinstance(arr, cp.ndarray):
        with cp.cuda.Device(int(arr.device.id)):
            return np.ascontiguousarray(cp.asnumpy(arr), dtype=dtype)
    return np.ascontiguousarray(np.asarray(arr), dtype=dtype)


def _validate_csr_shapes(
    cell_offsets: Any,
    cell_uv: Any,
    vis_data: Any,
    vis_times: Any,
) -> Tuple[int, int]:
    """Validate CSR buffer dimensions and return ``(m_active, nnz)``."""
    if cell_offsets.ndim != 1:
        raise ValueError(f"cell_offsets must be 1D; got shape {cell_offsets.shape}.")
    if cell_uv.ndim != 2 or int(cell_uv.shape[1]) != 2:
        raise ValueError(f"cell_uv must have shape (M, 2); got {cell_uv.shape}.")
    if vis_data.ndim != 1 or vis_times.ndim != 1:
        raise ValueError(
            f"vis_data and vis_times must be 1D; got {vis_data.shape} and {vis_times.shape}."
        )
    m_active = int(cell_uv.shape[0])
    if int(cell_offsets.shape[0]) != m_active + 1:
        raise ValueError(
            f"cell_offsets length ({cell_offsets.shape[0]}) must equal M + 1 ({m_active + 1})."
        )
    if int(vis_data.shape[0]) != int(vis_times.shape[0]):
        raise ValueError(
            f"vis_data length ({vis_data.shape[0]}) != vis_times length ({vis_times.shape[0]})."
        )
    return m_active, int(vis_data.shape[0])


def write_ragged_store_to_disk(
    store_dir: str,
    cell_offsets: Any,
    cell_uv: Any,
    vis_data: Any,
    vis_times: Any,
    attrs: Optional[Dict[str, Any]] = None,
) -> FastduccRaggedDGAStore:
    """
    Serialise host or GPU CSR buffers to ``store_dir`` and open a
    :class:`fastducc.ragged.RaggedDGAStore`.
    """
    if not store_dir:
        raise ValueError("store_dir must be a non-empty path when writing a RaggedDGAStore to disk.")
    os.makedirs(store_dir, exist_ok=True)
    h_off = _as_host_contiguous(cell_offsets, np.uint64)
    h_uv = _as_host_contiguous(cell_uv, np.int32)
    h_vis = _as_host_contiguous(vis_data, np.complex64)
    h_times = _as_host_contiguous(vis_times, np.float64)
    m_active, nnz = _validate_csr_shapes(h_off, h_uv, h_vis, h_times)

    h_off.tofile(os.path.join(store_dir, "cell_offsets.bin"))
    h_uv.tofile(os.path.join(store_dir, "cell_uv.bin"))
    h_vis.tofile(os.path.join(store_dir, "vis_data.bin"))
    h_times.tofile(os.path.join(store_dir, "vis_times.bin"))

    store_attrs = dict(attrs) if attrs else {}
    store_attrs.update(
        {
            "format": "ragged_csr",
            "m_total": int(m_active),
            "nnz_total": int(nnz),
        }
    )
    with open(os.path.join(store_dir, "attrs.json"), "w", encoding="utf-8") as fh:
        json.dump(store_attrs, fh, indent=2)
    return FastduccRaggedDGAStore.open(store_dir, mode="r")


def from_gpu_buffers(
    cell_offsets: Any,
    cell_uv: Any,
    vis_data: Any,
    vis_times: Any,
    attrs: Optional[Dict[str, Any]] = None,
    store_dir: str = "",
    copy_to_host: bool = False,
    device_id: Optional[int] = None,
    write_to_disk: bool = False,
    store_cls: Optional[type] = None,
) -> Any:
    """
    Construct a GPU or host ragged CSR store directly from CuPy (or host) buffers.

    Parameters
    ----------
    cell_offsets : array_like (uint64), shape (M + 1,)
        CSR row pointers per active UV cell.
    cell_uv : array_like (int32), shape (M, 2)
        Integer half-plane cell coordinates ``[v, u]``.
    vis_data : array_like (complex64), shape (NNZ,)
        Non-zero complex visibility samples.
    vis_times : array_like (float64), shape (NNZ,)
        Sample timestamps in seconds relative to observation start ``t0``.
    attrs : dict, optional
        Store metadata (``tsamp``, ``t0``, ``du``, ``dv``, ``npix_x``, ``npix_y``, ``nv``, ``nu``).
    store_dir : str, optional
        Optional directory label (and target directory when ``write_to_disk=True``).
    copy_to_host : bool, default False
        If False, wrap buffers zero-copy in VRAM as a :class:`GPURaggedDGAStore`.
        If True, copy arrays to host NumPy memory as a :class:`RaggedDGAStore`.
    device_id : int, optional
        Target CUDA device ID when ``copy_to_host=False``.
    write_to_disk : bool, default False
        When ``copy_to_host=True`` and ``store_dir`` is non-empty, serialise the
        binary CSR store files to ``store_dir``.
    store_cls : type, optional
        Optional host ``RaggedDGAStore`` class override when ``copy_to_host=True``.
    """
    store_attrs = dict(attrs) if attrs else {}
    dir_str = str(store_dir or "")

    if copy_to_host:
        h_off = _as_host_contiguous(cell_offsets, np.uint64)
        h_uv = _as_host_contiguous(cell_uv, np.int32)
        h_vis = _as_host_contiguous(vis_data, np.complex64)
        h_times = _as_host_contiguous(vis_times, np.float64)
        m_active, nnz = _validate_csr_shapes(h_off, h_uv, h_vis, h_times)
        store_attrs.setdefault("format", "ragged_csr")
        store_attrs["m_total"] = int(m_active)
        store_attrs["nnz_total"] = int(nnz)
        if write_to_disk and dir_str:
            written = write_ragged_store_to_disk(
                dir_str, h_off, h_uv, h_vis, h_times, attrs=store_attrs
            )
            if store_cls is not None and not isinstance(written, store_cls):
                return store_cls(
                    store_dir=written.store_dir,
                    attrs=dict(written.attrs),
                    cell_offsets=written.cell_offsets,
                    cell_uv=written.cell_uv,
                    vis_data=written.vis_data,
                    vis_times=written.vis_times,
                )
            return written
        target_cls = store_cls or FastduccRaggedDGAStore
        return target_cls(
            store_dir=dir_str,
            attrs=store_attrs,
            cell_offsets=h_off,
            cell_uv=h_uv,
            vis_data=h_vis,
            vis_times=h_times,
        )

    eff_dev = _infer_device_id(cell_offsets, cell_uv, vis_data, vis_times) if device_id is None else int(device_id)
    _require_cuda(eff_dev)
    with cp.cuda.Device(eff_dev):
        d_off = _as_gpu_zero_copy(cell_offsets, cp.uint64, eff_dev)
        d_uv = _as_gpu_zero_copy(cell_uv, cp.int32, eff_dev)
        d_vis = _as_gpu_zero_copy(vis_data, cp.complex64, eff_dev)
        d_times = _as_gpu_zero_copy(vis_times, cp.float64, eff_dev)
        m_active, nnz = _validate_csr_shapes(d_off, d_uv, d_vis, d_times)
        store_attrs.setdefault("format", "ragged_csr")
        store_attrs["m_total"] = int(m_active)
        store_attrs["nnz_total"] = int(nnz)

    gpu_cls = _get_gpu_store_cls()
    return gpu_cls(
        store_dir=dir_str,
        attrs=store_attrs,
        cell_offsets=d_off,
        cell_uv=d_uv,
        vis_data=d_vis,
        vis_times=d_times,
        device_id=eff_dev,
    )


def _is_csr_matrix(obj: Any) -> bool:
    """Return True if ``obj`` is a SciPy or CuPy CSR sparse matrix."""
    if HAS_CUPY_SPARSE and cupy_sparse is not None and isinstance(obj, cupy_sparse.csr_matrix):
        return True
    return hasattr(obj, "indptr") and hasattr(obj, "indices") and hasattr(obj, "data") and hasattr(obj, "shape")


def _extract_store_components(
    gpu_store: Any,
    *,
    cell_uv: Optional[Any] = None,
    dt: Optional[float] = None,
    attrs: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any, Any, Any, Dict[str, Any], str, int]:
    """
    Normalise a ``GPURaggedDGAStore``, ``RaggedDGAStore``, ``dict``, or
    ``cupyx.scipy.sparse.csr_matrix`` into raw CSR components
    ``(cell_offsets, cell_uv, vis_data, vis_times, attrs, store_dir, device_id)``.
    """
    if _is_csr_matrix(gpu_store):
        merged_attrs = dict(attrs or getattr(gpu_store, "attrs", {}) or {})
        eff_dt = float(merged_attrs.get("tsamp", 0.0) if dt is None else dt)
        if not math.isfinite(eff_dt) or eff_dt <= 0.0:
            raise ValueError(
                f"Positive sampling interval dt (or attrs['tsamp']) is required for csr_matrix inputs; got {eff_dt}."
            )
        uv_arr = cell_uv if cell_uv is not None else getattr(gpu_store, "cell_uv", None)
        if uv_arr is None:
            raise ValueError("cell_uv of shape (M_active, 2) must be provided alongside a csr_matrix.")
        dev_id = _infer_device_id(gpu_store.data, gpu_store.indptr, gpu_store.indices, uv_arr)
        off = gpu_store.indptr
        data = gpu_store.data
        idx = gpu_store.indices
        if HAS_CUPY and cp is not None and isinstance(idx, cp.ndarray):
            with cp.cuda.Device(dev_id):
                times = idx.astype(cp.float64) * eff_dt
        else:
            times = np.asarray(idx, dtype=np.float64) * eff_dt
        merged_attrs.setdefault("tsamp", eff_dt)
        merged_attrs.setdefault("total_planes", int(gpu_store.shape[1]))
        merged_attrs.setdefault("m_total", int(gpu_store.shape[0]))
        merged_attrs.setdefault("nnz_total", int(gpu_store.nnz))
        return off, uv_arr, data, times, merged_attrs, "", dev_id

    if isinstance(gpu_store, dict):
        merged_attrs = dict(gpu_store.get("attrs") or {})
        if attrs:
            merged_attrs.update(attrs)
        off = gpu_store.get("cell_offsets", gpu_store.get("d_cell_offsets"))
        uv_arr = cell_uv if cell_uv is not None else gpu_store.get("cell_uv", gpu_store.get("d_cell_uv"))
        data = gpu_store.get("vis_data", gpu_store.get("d_vis_data"))
        times = gpu_store.get("vis_times", gpu_store.get("d_vis_times"))
        store_dir = str(gpu_store.get("store_dir", "") or "")
        dev_id = int(gpu_store.get("device_id", _infer_device_id(off, uv_arr, data, times)))
    else:
        merged_attrs = dict(getattr(gpu_store, "attrs", {}) or {})
        if attrs:
            merged_attrs.update(attrs)
        off = getattr(gpu_store, "cell_offsets", getattr(gpu_store, "d_cell_offsets", None))
        uv_arr = cell_uv if cell_uv is not None else getattr(gpu_store, "cell_uv", getattr(gpu_store, "d_cell_uv", None))
        data = getattr(gpu_store, "vis_data", getattr(gpu_store, "d_vis_data", None))
        times = getattr(gpu_store, "vis_times", getattr(gpu_store, "d_vis_times", None))
        store_dir = str(getattr(gpu_store, "store_dir", "") or "")
        dev_id = int(getattr(gpu_store, "device_id", _infer_device_id(off, uv_arr, data, times)))

    if off is None or uv_arr is None or data is None or times is None:
        raise ValueError(
            "gpu_store must provide cell_offsets, cell_uv, vis_data, and vis_times."
        )
    if dt is not None:
        merged_attrs.setdefault("tsamp", float(dt))
    return off, uv_arr, data, times, merged_attrs, store_dir, dev_id


def gpu_store_to_host_ragged(
    gpu_store: Any,
    store_dir: str = "",
    *,
    cell_uv: Optional[Any] = None,
    dt: Optional[float] = None,
    attrs: Optional[Dict[str, Any]] = None,
    write_to_disk: bool = False,
) -> FastduccRaggedDGAStore:
    """
    Convert an in-VRAM :class:`GPURaggedDGAStore` (or ``dict`` / ``csr_matrix`` /
    host :class:`RaggedDGAStore`) to a host :class:`fastducc.ragged.RaggedDGAStore`.
    """
    off, uv_arr, data, times, merged_attrs, orig_dir, _ = _extract_store_components(
        gpu_store, cell_uv=cell_uv, dt=dt, attrs=attrs
    )
    eff_dir = str(store_dir if store_dir else orig_dir)
    return from_gpu_buffers(
        cell_offsets=off,
        cell_uv=uv_arr,
        vis_data=data,
        vis_times=times,
        attrs=merged_attrs,
        store_dir=eff_dir,
        copy_to_host=True,
        write_to_disk=bool(write_to_disk),
        store_cls=FastduccRaggedDGAStore,
    )


def build_gpu_csr_matrix(
    gpu_store: Any,
    total_planes: int,
    dt: float,
    validate: bool = False,
    *,
    cell_uv: Optional[Any] = None,
) -> Any:
    """
    Construct a ``cupyx.scipy.sparse.csr_matrix`` of shape ``(M_active, total_planes)``
    on GPU directly from an in-VRAM ``GPURaggedDGAStore`` or CSR buffer dict.

    Row ``m`` corresponds to ``cell_uv[m] = [v, u]``, column indices are
    ``t_idx = round(vis_times / dt)``, and ``indptr`` mirrors ``cell_offsets``
    (``int32`` when dimensions and NNZ fit, otherwise ``int64``).

    Parameters
    ----------
    gpu_store : GPURaggedDGAStore, RaggedDGAStore, dict, or csr_matrix
        Source CSR container.
    total_planes : int
        Total number of regularised time-grid planes (number of matrix columns).
    dt : float
        Nominal time sampling interval in seconds.
    validate : bool, default False
        If True, verify monotonic offsets, finite values, column bounds
        ``0 <= t_idx < total_planes``, and strictly increasing column indices per row.

    Returns
    -------
    csr : cupyx.scipy.sparse.csr_matrix (complex64)
        Sparse CSR matrix of shape ``(M_active, total_planes)``.
    """
    if not HAS_CUPY_SPARSE or cupy_sparse is None:
        raise RuntimeError("cupyx.scipy.sparse is required to build a GPU CSR matrix.")

    total_planes_i = int(total_planes)
    dt_f = float(dt)
    if total_planes_i < 0:
        raise ValueError(f"total_planes must be non-negative; got {total_planes}.")
    if not math.isfinite(dt_f) or dt_f <= 0.0:
        raise ValueError(f"dt must be finite and positive; got {dt}.")

    off, uv_arr, data, times, _, _, dev_id = _extract_store_components(
        gpu_store, cell_uv=cell_uv, dt=dt_f
    )
    _require_cuda(dev_id)

    with cp.cuda.Device(dev_id):
        d_off = _as_gpu_zero_copy(off, cp.uint64, dev_id)
        d_uv = _as_gpu_zero_copy(uv_arr, cp.int32, dev_id)
        d_data = _as_gpu_zero_copy(data, cp.complex64, dev_id)
        d_times = _as_gpu_zero_copy(times, cp.float64, dev_id)
        m_active, nnz = _validate_csr_shapes(d_off, d_uv, d_data, d_times)

        if validate:
            if int(d_off[0].get()) != 0:
                raise ValueError(f"cell_offsets[0] must be 0; got {int(d_off[0].get())}.")
            if d_off.size > 1 and not bool(cp.all(d_off[1:] >= d_off[:-1]).get()):
                raise ValueError("cell_offsets must be monotonically non-decreasing.")
            off_nnz = int(d_off[-1].get())
            if off_nnz != nnz:
                raise ValueError(f"cell_offsets[-1] ({off_nnz}) != vis_data length ({nnz}).")
            if nnz > 0:
                if not bool(cp.all(cp.isfinite(d_times)).get()):
                    raise ValueError("vis_times contains non-finite values.")
                if not bool((cp.all(cp.isfinite(d_data.real)) & cp.all(cp.isfinite(d_data.imag))).get()):
                    raise ValueError("vis_data contains non-finite values.")

        use_int32 = (
            nnz <= _INT32_MAX
            and m_active <= _INT32_MAX
            and total_planes_i <= _INT32_MAX
        )
        idx_dtype = cp.int32 if use_int32 else cp.int64

        if nnz == 0:
            d_indices = cp.empty(0, dtype=idx_dtype)
        else:
            d_t_round = cp.rint(d_times / dt_f)
            if validate:
                t_min = int(cp.min(d_t_round).get())
                t_max = int(cp.max(d_t_round).get())
                if t_min < 0 or t_max >= total_planes_i:
                    raise ValueError(
                        f"Computed time indices [{t_min}, {t_max}] out of range [0, {total_planes_i})."
                    )
            d_indices = d_t_round.astype(idx_dtype)
            del d_t_round

        if validate and nnz > 1 and m_active > 0:
            row_counts = d_off[1:] - d_off[:-1]
            if bool(cp.any(row_counts > 1).get()):
                h_off = cp.asnumpy(d_off)
                h_idx = cp.asnumpy(d_indices)
                for m_row in range(m_active):
                    s = int(h_off[m_row])
                    e = int(h_off[m_row + 1])
                    if e - s > 1 and np.any(np.diff(h_idx[s:e]) <= 0):
                        raise ValueError(
                            f"Non-monotonic or duplicate time indices in CSR row {m_row}."
                        )

        d_indptr = d_off.astype(idx_dtype, copy=False)
        csr = cupy_sparse.csr_matrix(
            (d_data, d_indices, d_indptr),
            shape=(m_active, total_planes_i),
            copy=False,
        )
        csr.has_canonical_format = True
        csr.has_sorted_indices = True
        return csr


def _resolve_grid_dims(
    attrs: Dict[str, Any],
    nx: Optional[int],
    ny: Optional[int],
    nv: Optional[int],
    nu: Optional[int],
) -> Tuple[int, int, int, int]:
    """Resolve image dimensions ``(nx, ny)`` and UV grid dimensions ``(nv, nu)``."""
    eff_nx = int(attrs.get("npix_x", 384) if nx is None else nx)
    eff_ny = int(attrs.get("npix_y", eff_nx) if ny is None else ny)
    eff_nv = int(attrs.get("nv", eff_ny) if nv is None else nv)
    eff_nu = int(attrs.get("nu", eff_nx // 2 + 1) if nu is None else nu)
    return eff_nx, eff_ny, eff_nv, eff_nu


def _run_stage234_gpu(
    d_offsets: "cp.ndarray",
    d_cell_uv: "cp.ndarray",
    d_vis_data: "cp.ndarray",
    d_vis_times: "cp.ndarray",
    *,
    device_id: int,
    nx: int,
    ny: int,
    nv: int,
    nu: int,
    Nf: int,
    df: float,
    kmin: int,
    nkeep: int,
    fkeep: np.ndarray,
    fmin_hz: float,
    fmax_hz: float,
    uv_threshold: float,
    detrend_uv: bool,
    detrend_min_samples: int,
    uv_block_mad_thresh: float,
    uv_mad_min_samples: int,
    epsilon: float,
    threshold_sigma: float,
    nharm: int,
    spatial_estimator: str,
    clip_sigma: float,
    use_local_threshold: bool,
    local_window_size: int,
    keep_top_k: Optional[int],
    gpu_freq_slab_size: int,
    gpu_c2r_batch_size: int,
    force_memmap: bool,
    scratch_dir: str,
    ms_base: str,
    dm_val: float,
    save_cube_path: Optional[str],
    cube_wcs: Optional[Any],
    extract_products: bool,
    cutout_size: int,
    save_fullframes: bool,
    inplace_preprocess: bool,
    gpu_store_to_release: Optional[Any],
) -> Dict[str, Any]:
    """Execute zero-copy Stage 2/3/4 directly on GPU buffers."""
    _ensure_periodicity_importable()
    from periodicity.c2r_imaging import C2RBatchWorkspace, synthesize_c2r_power_block_gpu
    from periodicity.gpu_stats import detect_nufft_harmonic_candidates_gpu
    from periodicity.ragged_dga import detrend_ragged_median_gpu, flag_ragged_chunk_mad_gpu
    from periodicity.run_nufft import (
        _extract_nufft_candidate_arrays,
        _extract_nufft_candidate_arrays_gpu,
        _fadvise_dontneed,
        _pwrite_all,
    )
    from periodicity.temporal_nufft_gpu import (
        evaluate_ragged_temporal_nufft_slab_gpu,
        prepare_temporal_nufft_plan_gpu,
    )

    m_total = int(d_cell_uv.shape[0])
    nnz = int(d_vis_data.size)
    if m_total == 0 or nnz == 0 or int(nkeep) <= 0:
        return {
            "detections": [],
            "max_snr_map": np.zeros((ny, nx), dtype=np.float32),
            "best_freq_map": np.zeros((ny, nx), dtype=np.float32),
            "cand_arrays": [] if extract_products else None,
            "timing": {
                "stage2_3_s": 0.0,
                "stage4_s": 0.0,
                "total_s": 0.0,
                "device": "cuda",
                "power_cube_location": "empty",
            },
        }

    t_start = time.time()
    need_preproc = (uv_threshold > 0.0) or bool(detrend_uv) or (uv_block_mad_thresh > 0.0)
    plan = None
    ws = None
    d_v_idx = None
    d_u_idx = None
    d_cos_buf = None
    d_sin_buf = None
    d_block_power = None
    d_power_cube = None
    mm_power = None
    power_mmap_path: Optional[str] = None
    power_fd: Optional[int] = None
    cand_arrays = None

    try:
        with cp.cuda.Device(device_id):
            work_vis = d_vis_data if (inplace_preprocess or not need_preproc) else d_vis_data.copy()

            if uv_threshold > 0.0 and work_vis.size > 0:
                thr_sq = np.float32(float(uv_threshold) * float(uv_threshold))
                mag_sq = work_vis.real * work_vis.real + work_vis.imag * work_vis.imag
                work_vis[mag_sq <= thr_sq] = 0.0 + 0.0j
                del mag_sq

            if detrend_uv and work_vis.size > 0:
                detrend_ragged_median_gpu(
                    work_vis,
                    d_offsets,
                    min_samples=int(detrend_min_samples),
                )

            if uv_block_mad_thresh > 0.0 and work_vis.size > 0:
                n_flagged = flag_ragged_chunk_mad_gpu(
                    work_vis,
                    d_offsets,
                    z_thresh=float(uv_block_mad_thresh),
                    min_samples=int(uv_mad_min_samples),
                )
                if n_flagged > 0:
                    logger.info(
                        "  [DM=%.2f] GPU MAD flagging zeroed %d/%d UV cells.",
                        dm_val,
                        n_flagged,
                        m_total,
                    )

            t_stage2_start = time.time()
            d_v_idx = cp.ascontiguousarray(d_cell_uv[:, 0], dtype=cp.int32)
            d_u_idx = cp.ascontiguousarray(d_cell_uv[:, 1], dtype=cp.int32)

            plan = prepare_temporal_nufft_plan_gpu(
                local_offsets=d_offsets,
                vis_data=work_vis,
                vis_times=d_vis_times,
                Nf=int(Nf),
                df=float(df),
                kmin=int(kmin),
                nkeep=int(nkeep),
                epsilon=float(epsilon),
                cache_dense=True,
            )
            del work_vis
            if gpu_store_to_release is not None and hasattr(gpu_store_to_release, "release"):
                gpu_store_to_release.release()
            cp.get_default_memory_pool().free_all_blocks()

            free_vram, _ = cp.cuda.Device(device_id).mem_info
            grid_nu = (nu - 1) * 2 if nu <= nv // 2 + 1 else nu
            bytes_per_c2r_item = max(1, 24 * int(nv) * int(grid_nu))
            max_c2r_by_vram = max(1, int(0.25 * free_vram) // bytes_per_c2r_item)
            c2r_batch = max(1, min(int(gpu_c2r_batch_size), max_c2r_by_vram, max(1, int(nkeep))))
            ws_est_bytes = c2r_batch * bytes_per_c2r_item

            bytes_per_freq_slab = max(1, 16 * max(1, m_total))
            req_slab = max(1, min(int(gpu_freq_slab_size), max(1, int(nkeep))))
            headroom = (
                max(256 * 1024 * 1024, int(0.15 * free_vram))
                + ws_est_bytes
                + min(req_slab * bytes_per_freq_slab, int(0.4 * free_vram))
            )
            power_cube_bytes = int(nkeep) * int(ny) * int(nx) * 4
            use_vram_cube = (
                (not bool(force_memmap))
                and (int(nkeep) > 0)
                and (power_cube_bytes <= max(0, free_vram - headroom))
            )

            if use_vram_cube:
                try:
                    d_power_cube = cp.empty((int(nkeep), ny, nx), dtype=cp.float32)
                except cp.cuda.memory.OutOfMemoryError:
                    logger.warning(
                        "  [DM=%.2f] VRAM allocation for power cube (%.2f GiB) exhausted memory; falling back to disk memmap.",
                        dm_val,
                        power_cube_bytes / (1024.0**3),
                    )
                    d_power_cube = None
                    cp.get_default_memory_pool().free_all_blocks()
                    use_vram_cube = False

            free_after, _ = cp.cuda.Device(device_id).mem_info
            slab_budget = max(bytes_per_freq_slab, int(0.4 * max(1, free_after - ws_est_bytes)))
            eff_slab_size = max(1, min(req_slab, slab_budget // bytes_per_freq_slab))
            ws_batch = max(1, min(c2r_batch, eff_slab_size))

            slice_bytes = int(ny) * int(nx) * 4
            synced_upto = 0
            if not use_vram_cube:
                os.makedirs(scratch_dir, exist_ok=True)
                power_mmap_path = os.path.join(
                    scratch_dir, f"{ms_base}_nufft_power_dm{dm_val:.2f}_{uuid.uuid4().hex[:8]}.mmap"
                )
                power_fd = os.open(power_mmap_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
                os.ftruncate(power_fd, int(nkeep) * slice_bytes)
                d_block_power = cp.empty((ws_batch, ny, nx), dtype=cp.float32)

            logger.info(
                "  [DM=%.2f] Stage 2/3 (GPU): Evaluating %d spin frequencies (slab=%d, c2r_batch=%d, power_cube=%s)...",
                dm_val,
                int(nkeep),
                eff_slab_size,
                ws_batch,
                "VRAM" if use_vram_cube else "memmap",
            )

            ws = C2RBatchWorkspace.create(
                max_batch=ws_batch,
                nv=nv,
                nu=nu,
                ny=ny,
                nx=nx,
                device_id=device_id,
                allocate_out=False,
            )
            d_cos_buf = cp.empty((eff_slab_size, m_total), dtype=cp.complex64)
            d_sin_buf = cp.empty((eff_slab_size, m_total), dtype=cp.complex64)

            for slab_off in range(0, int(nkeep), eff_slab_size):
                b_slab = min(eff_slab_size, int(nkeep) - slab_off)
                evaluate_ragged_temporal_nufft_slab_gpu(
                    plan=plan,
                    slab_offset=slab_off,
                    slab_size=b_slab,
                    out_cos=d_cos_buf[:b_slab],
                    out_sin=d_sin_buf[:b_slab],
                )
                for sub0 in range(0, b_slab, ws_batch):
                    sub1 = min(b_slab, sub0 + ws_batch)
                    b_sub = sub1 - sub0
                    g_start = slab_off + sub0
                    g_stop = slab_off + sub1
                    out_sub = d_power_cube[g_start:g_stop] if use_vram_cube else d_block_power[:b_sub]
                    synthesize_c2r_power_block_gpu(
                        v_cos_blk=d_cos_buf[sub0:sub1],
                        v_sin_blk=d_sin_buf[sub0:sub1],
                        v_idx=d_v_idx,
                        u_idx=d_u_idx,
                        nv=nv,
                        nu=nu,
                        ny=ny,
                        nx=nx,
                        out=out_sub,
                        workspace=ws,
                    )
                    if not use_vram_cube:
                        host_blk = cp.asnumpy(out_sub).astype(np.float32, copy=False)
                        _pwrite_all(power_fd, host_blk, g_start * slice_bytes)
                        del host_blk
                        if (g_stop * slice_bytes - synced_upto) >= _POWER_SYNC_BYTES or g_stop == int(nkeep):
                            os.fdatasync(power_fd)
                            _fadvise_dontneed(power_fd, synced_upto, g_stop * slice_bytes - synced_upto)
                            synced_upto = g_stop * slice_bytes

            if not use_vram_cube and power_fd is not None:
                os.fdatasync(power_fd)
                _fadvise_dontneed(power_fd, 0, 0)
                os.close(power_fd)
                power_fd = None
                mm_power = np.memmap(power_mmap_path, mode="r", dtype=np.float32, shape=(int(nkeep), ny, nx))

            t_stage23 = time.time() - t_stage2_start

            ws.release()
            ws = None
            d_cos_buf = None
            d_sin_buf = None
            d_v_idx = None
            d_u_idx = None
            d_block_power = None
            plan.d_offsets = None
            plan.d_vis_data = None
            plan.d_vis_times = None
            plan.d_empty_idx = None
            plan.d_sparse_idx = None
            plan.d_dense_idx = None
            plan.d_inv_deapod = None
            plan.d_dense_cos = None
            plan.d_dense_sin = None
            plan = None
            cp.get_default_memory_pool().free_all_blocks()

            t_stage4_start = time.time()
            cube_for_stage4 = d_power_cube if use_vram_cube else mm_power
            dets, max_snr_map, best_freq_map = detect_nufft_harmonic_candidates_gpu(
                cube_yx=cube_for_stage4,
                fkeep=fkeep,
                kmin=int(kmin),
                fmin_hz=float(fmin_hz),
                fmax_hz=float(fmax_hz),
                threshold_sigma=float(threshold_sigma),
                nharm=int(nharm),
                spatial_estimator=str(spatial_estimator),
                clip_sigma=float(clip_sigma),
                use_local_threshold=bool(use_local_threshold),
                local_window_size=int(local_window_size),
                keep_top_k=keep_top_k,
            )
            t_stage4 = time.time() - t_stage4_start

            if save_cube_path:
                _save_power_cube_fits(
                    save_cube_path=save_cube_path,
                    cube_data=(
                        cp.asnumpy(d_power_cube).astype(np.float32, copy=False)
                        if use_vram_cube
                        else np.asarray(mm_power, dtype=np.float32).copy()
                    ),
                    cube_wcs=cube_wcs,
                    fkeep=fkeep,
                    df=df,
                    fmin_hz=fmin_hz,
                    fmax_hz=fmax_hz,
                    dm_val=dm_val,
                )

            if extract_products and dets:
                if use_vram_cube:
                    cand_arrays = _extract_nufft_candidate_arrays_gpu(
                        cands=dets,
                        d_power_cube=d_power_cube,
                        fkeep=fkeep,
                        kmin=int(kmin),
                        cutout_size=int(cutout_size),
                        save_fullframes=bool(save_fullframes),
                    )
                else:
                    cand_arrays = _extract_nufft_candidate_arrays(
                        cands=dets,
                        cube_yx=mm_power,
                        fkeep=fkeep,
                        kmin=int(kmin),
                        cutout_size=int(cutout_size),
                        save_fullframes=bool(save_fullframes),
                        cache_path=power_mmap_path,
                    )
            elif extract_products:
                cand_arrays = []

            return {
                "detections": dets,
                "max_snr_map": max_snr_map,
                "best_freq_map": best_freq_map,
                "cand_arrays": cand_arrays,
                "timing": {
                    "stage2_3_s": float(t_stage23),
                    "stage4_s": float(t_stage4),
                    "total_s": float(time.time() - t_start),
                    "device": "cuda",
                    "power_cube_location": "VRAM" if use_vram_cube else "memmap",
                },
            }
    finally:
        if power_fd is not None:
            try:
                os.close(power_fd)
            except OSError:
                pass
        if ws is not None:
            ws.release()
        d_cos_buf = None
        d_sin_buf = None
        d_v_idx = None
        d_u_idx = None
        d_block_power = None
        d_power_cube = None
        plan = None
        if mm_power is not None:
            if getattr(mm_power, "_mmap", None) is not None:
                try:
                    mm_power._mmap.close()
                except Exception:
                    pass
            mm_power = None
        if power_mmap_path and os.path.exists(power_mmap_path):
            try:
                os.remove(power_mmap_path)
            except OSError:
                pass
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        gc.collect()


def _run_stage234_cpu(
    host_store: FastduccRaggedDGAStore,
    *,
    nx: int,
    ny: int,
    nv: int,
    nu: int,
    Nf: int,
    df: float,
    kmin: int,
    nkeep: int,
    fkeep: np.ndarray,
    fmin_hz: float,
    fmax_hz: float,
    uv_threshold: float,
    detrend_uv: bool,
    detrend_min_samples: int,
    uv_block_mad_thresh: float,
    uv_mad_min_samples: int,
    epsilon: float,
    threshold_sigma: float,
    nharm: int,
    spatial_estimator: str,
    clip_sigma: float,
    use_local_threshold: bool,
    local_window_size: int,
    keep_top_k: Optional[int],
    freq_block_size: int,
    scratch_dir: str,
    ms_base: str,
    dm_val: float,
    save_cube_path: Optional[str],
    cube_wcs: Optional[Any],
    extract_products: bool,
    cutout_size: int,
    save_fullframes: bool,
    nthreads: int,
) -> Dict[str, Any]:
    """Execute CPU fallback Stage 2/3/4 from a host :class:`RaggedDGAStore`."""
    _ensure_periodicity_importable()
    import ducc0.fft as dfft
    from periodicity.c2r_imaging import _scatter_visibilities
    from periodicity.ragged_dga import detrend_ragged_median_numba, flag_ragged_chunk_mad
    from periodicity.run_nufft import (
        _extract_nufft_candidate_arrays,
        _fadvise_dontneed,
        _pwrite_all,
        detect_nufft_harmonic_candidates,
    )
    from periodicity.temporal_nufft import evaluate_ragged_temporal_nufft_batch

    m_total = host_store.m_total
    nnz = host_store.nnz
    if m_total == 0 or nnz == 0 or int(nkeep) <= 0:
        return {
            "detections": [],
            "max_snr_map": np.zeros((ny, nx), dtype=np.float32),
            "best_freq_map": np.zeros((ny, nx), dtype=np.float32),
            "cand_arrays": [] if extract_products else None,
            "timing": {
                "stage2_3_s": 0.0,
                "stage4_s": 0.0,
                "total_s": 0.0,
                "device": "cpu",
                "power_cube_location": "empty",
            },
        }

    t_start = time.time()
    t_stage2_start = time.time()
    local_offsets = np.ascontiguousarray(host_store.cell_offsets, dtype=np.uint64)
    vis_data = np.array(host_store.vis_data, dtype=np.complex64, copy=True)
    vis_times = np.ascontiguousarray(host_store.vis_times, dtype=np.float64)
    cell_uv = np.ascontiguousarray(host_store.cell_uv, dtype=np.int64)

    if uv_threshold > 0.0 and vis_data.size > 0:
        thr_sq = float(uv_threshold) * float(uv_threshold)
        mag_sq = vis_data.real * vis_data.real + vis_data.imag * vis_data.imag
        vis_data[mag_sq <= thr_sq] = 0.0 + 0.0j

    if detrend_uv and vis_data.size > 0:
        detrend_ragged_median_numba(vis_data, local_offsets, min_samples=int(detrend_min_samples))

    if uv_block_mad_thresh > 0.0 and vis_data.size > 0:
        flag_ragged_chunk_mad(
            vis_data,
            local_offsets,
            z_thresh=float(uv_block_mad_thresh),
            min_samples=int(uv_mad_min_samples),
        )

    v_cos_t, v_sin_t = evaluate_ragged_temporal_nufft_batch(
        local_offsets=local_offsets,
        vis_data=vis_data,
        vis_times=vis_times,
        Nf=int(Nf),
        df=float(df),
        kmin=int(kmin),
        nkeep=int(nkeep),
        epsilon=float(epsilon),
        nthreads=max(1, int(nthreads)),
    )
    del vis_data, vis_times, local_offsets

    grid_nu = (nu - 1) * 2 if nu <= nv // 2 + 1 else nu
    u_half = nu if nu <= nv // 2 + 1 else nu // 2 + 1
    half_ny = ny // 2
    half_nx = nx // 2
    idx_y = np.concatenate([np.arange(nv - half_ny, nv), np.arange(0, ny - half_ny)]).astype(np.int64)
    idx_x = np.concatenate([np.arange(grid_nu - half_nx, grid_nu), np.arange(0, nx - half_nx)]).astype(np.int64)

    os.makedirs(scratch_dir, exist_ok=True)
    power_mmap_path = os.path.join(
        scratch_dir, f"{ms_base}_nufft_power_dm{dm_val:.2f}_{uuid.uuid4().hex[:8]}.mmap"
    )
    slice_bytes = int(ny) * int(nx) * 4
    power_fd = os.open(power_mmap_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    mm_power = None
    cand_arrays = None

    try:
        os.ftruncate(power_fd, int(nkeep) * slice_bytes)
        synced_upto = 0
        c2r_threads = max(1, min(4, int(nthreads)))
        fblk = max(1, int(freq_block_size))

        for ks in range(0, int(nkeep), fblk):
            ke = min(ks + fblk, int(nkeep))
            b_curr = ke - ks
            grid_cos = np.zeros((b_curr, nv, u_half), dtype=np.complex64)
            grid_sin = np.zeros((b_curr, nv, u_half), dtype=np.complex64)
            _scatter_visibilities(grid_cos, cell_uv[:, 0], cell_uv[:, 1], v_cos_t[ks:ke])
            _scatter_visibilities(grid_sin, cell_uv[:, 0], cell_uv[:, 1], v_sin_t[ks:ke])

            dirty_cos = np.empty((b_curr, nv, grid_nu), dtype=np.float32)
            dirty_sin = np.empty((b_curr, nv, grid_nu), dtype=np.float32)
            dfft.c2r(
                grid_cos, axes=(1, 2), lastsize=grid_nu, forward=False, inorm=2, out=dirty_cos, nthreads=c2r_threads
            )
            dfft.c2r(
                grid_sin, axes=(1, 2), lastsize=grid_nu, forward=False, inorm=2, out=dirty_sin, nthreads=c2r_threads
            )
            crop_cos = dirty_cos[:, idx_y[:, None], idx_x[None, :]]
            crop_sin = dirty_sin[:, idx_y[:, None], idx_x[None, :]]
            power_blk = (crop_cos * crop_cos + crop_sin * crop_sin).astype(np.float32, copy=False)
            _pwrite_all(power_fd, power_blk, ks * slice_bytes)
            del grid_cos, grid_sin, dirty_cos, dirty_sin, crop_cos, crop_sin, power_blk
            if (ke * slice_bytes - synced_upto) >= _POWER_SYNC_BYTES or ke == int(nkeep):
                os.fdatasync(power_fd)
                _fadvise_dontneed(power_fd, synced_upto, ke * slice_bytes - synced_upto)
                synced_upto = ke * slice_bytes

        del v_cos_t, v_sin_t
        os.fdatasync(power_fd)
        _fadvise_dontneed(power_fd, 0, 0)
        os.close(power_fd)
        power_fd = None
        t_stage23 = time.time() - t_stage2_start

        mm_power = np.memmap(power_mmap_path, mode="r", dtype=np.float32, shape=(int(nkeep), ny, nx))
        t_stage4_start = time.time()
        dets, max_snr_map, best_freq_map = detect_nufft_harmonic_candidates(
            cube_yx=mm_power,
            fkeep=fkeep,
            kmin=int(kmin),
            fmin_hz=float(fmin_hz),
            fmax_hz=float(fmax_hz),
            threshold_sigma=float(threshold_sigma),
            nharm=int(nharm),
            spatial_estimator=str(spatial_estimator),
            clip_sigma=float(clip_sigma),
            use_local_threshold=bool(use_local_threshold),
            local_window_size=int(local_window_size),
            keep_top_k=keep_top_k,
            cache_path=power_mmap_path,
        )
        t_stage4 = time.time() - t_stage4_start

        if save_cube_path:
            _save_power_cube_fits(
                save_cube_path=save_cube_path,
                cube_data=np.asarray(mm_power, dtype=np.float32).copy(),
                cube_wcs=cube_wcs,
                fkeep=fkeep,
                df=df,
                fmin_hz=fmin_hz,
                fmax_hz=fmax_hz,
                dm_val=dm_val,
            )

        if extract_products and dets:
            cand_arrays = _extract_nufft_candidate_arrays(
                cands=dets,
                cube_yx=mm_power,
                fkeep=fkeep,
                kmin=int(kmin),
                cutout_size=int(cutout_size),
                save_fullframes=bool(save_fullframes),
                cache_path=power_mmap_path,
            )
        elif extract_products:
            cand_arrays = []

        return {
            "detections": dets,
            "max_snr_map": max_snr_map,
            "best_freq_map": best_freq_map,
            "cand_arrays": cand_arrays,
            "timing": {
                "stage2_3_s": float(t_stage23),
                "stage4_s": float(t_stage4),
                "total_s": float(time.time() - t_start),
                "device": "cpu",
                "power_cube_location": "memmap",
            },
        }
    finally:
        if power_fd is not None:
            try:
                os.close(power_fd)
            except OSError:
                pass
        if mm_power is not None:
            if getattr(mm_power, "_mmap", None) is not None:
                try:
                    mm_power._mmap.close()
                except Exception:
                    pass
            mm_power = None
        if os.path.exists(power_mmap_path):
            try:
                os.remove(power_mmap_path)
            except OSError:
                pass
        gc.collect()


def _save_power_cube_fits(
    *,
    save_cube_path: str,
    cube_data: np.ndarray,
    cube_wcs: Optional[Any],
    fkeep: np.ndarray,
    df: float,
    fmin_hz: float,
    fmax_hz: float,
    dm_val: float,
) -> None:
    """Atomically serialise a 3D spin-frequency power cube FITS image."""
    from astropy.wcs import WCS
    from periodicity.io_utils import atomic_write_fits

    w3 = WCS(naxis=3)
    if cube_wcs is not None:
        for ax_i in (0, 1):
            w3.wcs.ctype[ax_i] = cube_wcs.wcs.ctype[ax_i]
            w3.wcs.cunit[ax_i] = cube_wcs.wcs.cunit[ax_i]
            w3.wcs.crpix[ax_i] = cube_wcs.wcs.crpix[ax_i]
            w3.wcs.crval[ax_i] = cube_wcs.wcs.crval[ax_i]
            w3.wcs.cdelt[ax_i] = cube_wcs.wcs.cdelt[ax_i]
    w3.wcs.ctype[2] = "FREQ"
    w3.wcs.cunit[2] = "Hz"
    w3.wcs.crpix[2] = 1.0
    w3.wcs.crval[2] = float(fkeep[0]) if len(fkeep) > 0 else 0.0
    w3.wcs.cdelt[2] = float(df)

    hdr3 = w3.to_header()
    hdr3["BUNIT"] = "arb"
    hdr3["FMIN_HZ"] = float(fmin_hz)
    hdr3["FMAX_HZ"] = float(fmax_hz)
    hdr3["DM"] = float(dm_val)
    atomic_write_fits(save_cube_path, cube_data, header=hdr3)
    logger.info("  Wrote 3D power cube -> %s", save_cube_path)


def run_gpu_nufft_from_store(
    gpu_store: Any,
    *,
    Nf: int,
    df: float,
    kmin: int,
    nkeep: int,
    fkeep: np.ndarray,
    fmin_hz: float,
    fmax_hz: float,
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    nv: Optional[int] = None,
    nu: Optional[int] = None,
    cell_uv: Optional[Any] = None,
    dt: Optional[float] = None,
    attrs: Optional[Dict[str, Any]] = None,
    uv_threshold: float = 1e-4,
    detrend_uv: bool = False,
    detrend_min_samples: int = 5,
    uv_block_mad_thresh: float = 12.0,
    uv_mad_min_samples: int = 20,
    epsilon: float = 1e-5,
    threshold_sigma: float = 7.0,
    nharm: int = 8,
    spatial_estimator: str = "clipped_rms",
    clip_sigma: float = 3.0,
    use_local_threshold: bool = True,
    local_window_size: int = 64,
    keep_top_k: Optional[int] = 200,
    gpu_freq_slab_size: int = 2048,
    gpu_c2r_batch_size: int = 256,
    freq_block_size: int = 256,
    force_memmap: bool = False,
    scratch_dir: Optional[str] = None,
    ms_base: str = "direct_raster",
    dm_val: float = 0.0,
    save_cube_path: Optional[str] = None,
    cube_wcs: Optional[Any] = None,
    extract_products: bool = False,
    cutout_size: int = 32,
    save_fullframes: bool = True,
    device: str = "auto",
    nthreads: int = 1,
    inplace_preprocess: bool = True,
    release_store: bool = False,
) -> Dict[str, Any]:
    """
    Execute zero-copy Stage 2 (temporal NUFFT), Stage 3 (batched 2D C2R power
    synthesis), and Stage 4 (harmonic summing & spatial peak detection) directly
    from an in-VRAM ``GPURaggedDGAStore``, ``dict``, ``csr_matrix``, or host
    ``RaggedDGAStore``.

    Supports ``device='cuda'``, ``device='cpu'``, and ``device='auto'`` (which
    automatically falls back to CPU if CUDA is unavailable).
    """
    eff_device = resolve_device(device)
    off, uv_arr, data, times, merged_attrs, _, dev_id = _extract_store_components(
        gpu_store, cell_uv=cell_uv, dt=dt, attrs=attrs
    )
    eff_nx, eff_ny, eff_nv, eff_nu = _resolve_grid_dims(merged_attrs, nx, ny, nv, nu)
    eff_scratch = scratch_dir or os.environ.get("JOBFS") or os.getcwd()

    if eff_device == "cuda":
        with cp.cuda.Device(dev_id):
            d_off = _as_gpu_zero_copy(off, cp.uint64, dev_id)
            d_uv = _as_gpu_zero_copy(uv_arr, cp.int32, dev_id)
            d_data = _as_gpu_zero_copy(data, cp.complex64, dev_id)
            d_times = _as_gpu_zero_copy(times, cp.float64, dev_id)
        return _run_stage234_gpu(
            d_offsets=d_off,
            d_cell_uv=d_uv,
            d_vis_data=d_data,
            d_vis_times=d_times,
            device_id=dev_id,
            nx=eff_nx,
            ny=eff_ny,
            nv=eff_nv,
            nu=eff_nu,
            Nf=int(Nf),
            df=float(df),
            kmin=int(kmin),
            nkeep=int(nkeep),
            fkeep=np.asarray(fkeep, dtype=np.float64),
            fmin_hz=float(fmin_hz),
            fmax_hz=float(fmax_hz),
            uv_threshold=float(uv_threshold),
            detrend_uv=bool(detrend_uv),
            detrend_min_samples=int(detrend_min_samples),
            uv_block_mad_thresh=float(uv_block_mad_thresh),
            uv_mad_min_samples=int(uv_mad_min_samples),
            epsilon=float(epsilon),
            threshold_sigma=float(threshold_sigma),
            nharm=int(nharm),
            spatial_estimator=str(spatial_estimator),
            clip_sigma=float(clip_sigma),
            use_local_threshold=bool(use_local_threshold),
            local_window_size=int(local_window_size),
            keep_top_k=keep_top_k,
            gpu_freq_slab_size=int(gpu_freq_slab_size),
            gpu_c2r_batch_size=int(gpu_c2r_batch_size),
            force_memmap=bool(force_memmap),
            scratch_dir=eff_scratch,
            ms_base=str(ms_base),
            dm_val=float(dm_val),
            save_cube_path=save_cube_path,
            cube_wcs=cube_wcs,
            extract_products=bool(extract_products),
            cutout_size=int(cutout_size),
            save_fullframes=bool(save_fullframes),
            inplace_preprocess=bool(inplace_preprocess),
            gpu_store_to_release=gpu_store if release_store else None,
        )

    host_store = gpu_store_to_host_ragged(
        gpu_store, cell_uv=cell_uv, dt=dt, attrs=merged_attrs, write_to_disk=False
    )
    if release_store and hasattr(gpu_store, "release"):
        gpu_store.release()
    return _run_stage234_cpu(
        host_store=host_store,
        nx=eff_nx,
        ny=eff_ny,
        nv=eff_nv,
        nu=eff_nu,
        Nf=int(Nf),
        df=float(df),
        kmin=int(kmin),
        nkeep=int(nkeep),
        fkeep=np.asarray(fkeep, dtype=np.float64),
        fmin_hz=float(fmin_hz),
        fmax_hz=float(fmax_hz),
        uv_threshold=float(uv_threshold),
        detrend_uv=bool(detrend_uv),
        detrend_min_samples=int(detrend_min_samples),
        uv_block_mad_thresh=float(uv_block_mad_thresh),
        uv_mad_min_samples=int(uv_mad_min_samples),
        epsilon=float(epsilon),
        threshold_sigma=float(threshold_sigma),
        nharm=int(nharm),
        spatial_estimator=str(spatial_estimator),
        clip_sigma=float(clip_sigma),
        use_local_threshold=bool(use_local_threshold),
        local_window_size=int(local_window_size),
        keep_top_k=keep_top_k,
        freq_block_size=int(freq_block_size),
        scratch_dir=eff_scratch,
        ms_base=str(ms_base),
        dm_val=float(dm_val),
        save_cube_path=save_cube_path,
        cube_wcs=cube_wcs,
        extract_products=bool(extract_products),
        cutout_size=int(cutout_size),
        save_fullframes=bool(save_fullframes),
        nthreads=int(nthreads),
    )


__all__ = [
    "is_cuda_available",
    "resolve_device",
    "write_ragged_store_to_disk",
    "from_gpu_buffers",
    "gpu_store_to_host_ragged",
    "build_gpu_csr_matrix",
    "run_gpu_nufft_from_store",
]
