"""Tests for fastducc.prune.prune_direct_dga_store."""

import os
import tempfile

import numpy as np
import pytest
import zarr

from fastducc.prune import prune_direct_dga_store, _OLD_DGA, _SCRATCH_DGA, _SCRATCH_UV


def _make_store(path, m=1300, nt=777, frac_active=0.5, seed=0, row_chunk=128, t_chunk=64):
    rng = np.random.default_rng(seed)
    active = rng.random(m) < frac_active
    data = np.zeros((m, nt), dtype=np.complex64)
    n_act = int(active.sum())
    data[active] = (rng.normal(size=(n_act, nt)) + 1j * rng.normal(size=(n_act, nt))).astype(np.complex64)
    # Sparse-in-time active cells and sub-threshold noise in "inactive" cells
    sparse_rows = np.flatnonzero(active)[:5]
    data[sparse_rows] = 0
    data[sparse_rows, rng.integers(0, nt, size=len(sparse_rows))] = 1.0 + 0.5j
    data[~active] = 1e-6
    uv = np.stack([np.arange(m) % 97, np.arange(m) // 97], axis=1).astype(np.int64)

    root = zarr.open_group(path, mode="w")
    root.create_array("dga", shape=data.shape, chunks=(row_chunk, t_chunk), dtype=np.complex64)
    root.create_array("uv", shape=uv.shape, chunks=(m, 2), dtype=np.int64)
    root["dga"][:] = data
    root["uv"][:] = uv
    root.attrs.update({"format": "direct_dga", "m_total": m, "total_planes": nt})
    return data, uv, active


@pytest.mark.parametrize("n_threads", [1, 4])
@pytest.mark.parametrize("chunk_cells", [100, 4096])
def test_prune_compacts_and_keeps_uv_synced(n_threads, chunk_cells):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "s.zarr")
        data, uv, active = _make_store(path)
        stats = prune_direct_dga_store(path, threshold=1e-4, chunk_cells=chunk_cells, n_threads=n_threads)

        assert stats["m_original"] == data.shape[0]
        assert stats["m_pruned"] == int(active.sum())
        root = zarr.open_group(path, mode="r")
        assert root["dga"].shape == (int(active.sum()), data.shape[1])
        np.testing.assert_array_equal(root["dga"][:], data[active])
        np.testing.assert_array_equal(root["uv"][:], uv[active])
        assert root.attrs["pruned"] is True
        assert root.attrs["m_total"] == int(active.sum())
        assert root.attrs["original_m_padded"] == data.shape[0]
        assert root.attrs["format"] == "direct_dga"
        assert sorted(os.listdir(path)) == ["dga", "uv", "zarr.json"]

        # Idempotent
        again = prune_direct_dga_store(path, threshold=1e-4)
        assert again["skipped"] is True
        np.testing.assert_array_equal(zarr.open_group(path, mode="r")["uv"][:], uv[active])


def test_prune_all_inactive_and_all_active():
    with tempfile.TemporaryDirectory() as td:
        p0 = os.path.join(td, "none.zarr")
        data, _, _ = _make_store(p0, frac_active=0.0)
        st = prune_direct_dga_store(p0, n_threads=2)
        assert st["m_pruned"] == 0
        assert zarr.open_group(p0, mode="r")["dga"].shape == (0, data.shape[1])

        p1 = os.path.join(td, "all.zarr")
        data, uv, _ = _make_store(p1, frac_active=1.01)
        st = prune_direct_dga_store(p1, n_threads=2)
        assert st["m_pruned"] == data.shape[0]
        np.testing.assert_array_equal(zarr.open_group(p1, mode="r")["dga"][:], data)


def test_prune_recovers_interrupted_swap():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "s.zarr")
        data, uv, active = _make_store(path)
        # Build the expected scratch arrays, then simulate a crash after the first rename
        root = zarr.open_group(path, mode="r+")
        root.create_array(_SCRATCH_DGA, shape=(int(active.sum()), data.shape[1]), chunks=(128, 64), dtype=np.complex64)
        root[_SCRATCH_DGA][:] = data[active]
        root.create_array(_SCRATCH_UV, shape=(int(active.sum()), 2), chunks=(int(active.sum()), 2), dtype=np.int64)
        root[_SCRATCH_UV][:] = uv[active]
        os.rename(os.path.join(path, "dga"), os.path.join(path, _OLD_DGA))
        os.rename(os.path.join(path, _SCRATCH_DGA), os.path.join(path, "dga"))

        st = prune_direct_dga_store(path)
        assert st["skipped"] is True
        root = zarr.open_group(path, mode="r")
        np.testing.assert_array_equal(root["dga"][:], data[active])
        np.testing.assert_array_equal(root["uv"][:], uv[active])
        assert root.attrs["original_m_padded"] == data.shape[0]
        assert sorted(os.listdir(path)) == ["dga", "uv", "zarr.json"]


def test_prune_with_dask_client():
    distributed = pytest.importorskip("dask.distributed")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "s.zarr")
        data, uv, active = _make_store(path, m=2100)
        with distributed.LocalCluster(n_workers=2, threads_per_worker=1, processes=False,
                                      dashboard_address=None) as cl, distributed.Client(cl) as client:
            st = prune_direct_dga_store(path, chunk_cells=256, client=client)
        assert st["m_pruned"] == int(active.sum())
        root = zarr.open_group(path, mode="r")
        np.testing.assert_array_equal(root["dga"][:], data[active])
        np.testing.assert_array_equal(root["uv"][:], uv[active])
