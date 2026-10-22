# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord


def _connected_components_from_pairs(n: int, i_idx, j_idx) -> List[List[int]]:
    """Compute connected components from undirected edge lists."""
    adj: List[List[int]] = [[] for _ in range(n)]
    for a, b in zip(i_idx, j_idx):
        a = int(a)
        b = int(b)
        if a == b:
            continue
        adj[a].append(b)
        adj[b].append(a)

    seen = [False] * n
    comps: List[List[int]] = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp = [s]
        while stack:
            v = stack.pop()
            for w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
                    comp.append(w)
        comps.append(comp)
    return comps


def _time_groups(times: np.ndarray, tol_s: float) -> List[Tuple[int, int]]:
    """Return [(i0,i1), ...] contiguous groups where (times[i]-times[i0]) <= tol_s."""
    if times.size == 0:
        return []
    order = np.argsort(times)
    times_sorted = times[order]
    groups = []
    i0 = 0
    N = times_sorted.size
    while i0 < N:
        t0 = times_sorted[i0]
        i1 = i0 + 1
        while i1 < N and (times_sorted[i1] - t0) <= tol_s:
            i1 += 1
        idxs = order[i0:i1]
        groups.append((int(idxs.min()), int(idxs.max()) + 1))
        i0 = i1
    return groups


def _cluster_indices_by_sky(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    seplimit_arcsec: float
) -> List[List[int]]:
    """
    Return list of components (each a list of row indices) by clustering positions
    within seplimit_arcsec.
    """
    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    i_idx, j_idx, _, _ = coords.search_around_sky(coords, seplimit=seplimit_arcsec * u.arcsec)
    comps = _connected_components_from_pairs(len(ra_deg), i_idx, j_idx)
    return comps
