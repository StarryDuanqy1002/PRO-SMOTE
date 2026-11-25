from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils.validation import check_array

@dataclass
class SNNGeometry:
    distances: np.ndarray
    order: np.ndarray
    neighbors: np.ndarray
    shared_counts: dict
    rho: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    effective_k: int

@dataclass
class SNNPartition:
    labels: np.ndarray
    centers: np.ndarray
    geometry: SNNGeometry

def snn_geometry(X, k=5, memory_limit_mb=1024):
    X = check_array(X, dtype=float, ensure_min_samples=1)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError('SNN-DPC k must be a positive integer.')
    if not np.isfinite(memory_limit_mb) or memory_limit_mb <= 0:
        raise ValueError('memory_limit_mb must be finite and positive.')
    n = len(X)
    if 32 * n * n / 1024 ** 2 > memory_limit_mb:
        raise MemoryError('SD-CSMOTE SNN-DPC pairwise memory limit exceeded.')
    k = min(int(k), n)
    distances = squareform(pdist(X))
    np.fill_diagonal(distances, -1.0)
    order = np.argsort(distances, axis=1, kind='stable')
    np.fill_diagonal(distances, 0.0)
    neighbors = order[:, :k]
    sets = [set(row.tolist()) for row in neighbors]
    similarities = [[] for _ in range(n)]
    shared_counts = {}
    for i in range(n):
        for j in neighbors[i]:
            j = int(j)
            if i == j:
                continue
            shared = sets[i] & sets[j]
            shared_counts[i, j] = len(shared)
            if j < i and i in sets[j]:
                ids = np.array(sorted(shared), dtype=int)
                denominator = float((distances[i, ids] + distances[j, ids]).sum())
                similarity = len(shared) ** 2 / denominator if denominator > 0 else 0.0
                similarities[i].append(similarity)
                similarities[j].append(similarity)
    rho = np.array([sum(sorted(row)[-k:]) for row in similarities], dtype=float)
    neighbor_sums = distances[np.arange(n)[:, None], neighbors].sum(axis=1)
    delta = np.zeros(n, dtype=float)
    maxima = []
    for i in range(n):
        higher = np.flatnonzero(rho > rho[i])
        if len(higher):
            delta[i] = np.min(distances[i, higher] * (neighbor_sums[i] + neighbor_sums[higher]))
        else:
            maxima.append(i)
    delta[maxima] = delta.max()
    return SNNGeometry(distances, order, neighbors, shared_counts, rho, delta, rho * delta, k)

def allocate_points(geometry, centers):
    n = len(geometry.rho)
    raw = np.asarray(centers)
    if raw.ndim != 1 or not len(raw) or raw.dtype.kind not in 'iu':
        raise ValueError('Center indices must be a nonempty 1D integer sequence.')
    centers = np.sort(raw.astype(int))
    if len(np.unique(centers)) != len(centers) or centers.min() < 0 or centers.max() >= n:
        raise ValueError('Centers must be distinct valid minority-training row indices.')
    labels = np.full(n, -1, dtype=int)
    labels[centers] = np.arange(len(centers))
    queue = deque(centers.tolist())
    while queue:
        i = queue.popleft()
        for j in geometry.neighbors[i]:
            if labels[j] < 0 and geometry.shared_counts.get((i, int(j)), 0) >= geometry.effective_k / 2:
                labels[j] = labels[i]
                queue.append(int(j))
    width = geometry.effective_k
    while np.any(labels < 0):
        remaining = np.flatnonzero(labels < 0)
        votes = np.zeros((len(remaining), len(centers)), dtype=int)
        for pos, i in enumerate(remaining):
            assigned = labels[geometry.order[i, :width]]
            votes[pos] = np.bincount(assigned[assigned >= 0], minlength=len(centers))
        most = votes.max()
        if most:
            rows = votes.max(axis=1) == most
            labels[remaining[rows]] = votes[rows].argmax(axis=1)
        elif width < n:
            width += 1
        else:
            raise RuntimeError('SNN-DPC allocation failed despite valid centers.')
    return SNNPartition(labels, centers, geometry)

def snn_dpc(X, k=5, n_clusters=2, center_indices=None, memory_limit_mb=1024):
    geometry = snn_geometry(X, k, memory_limit_mb)
    if center_indices is None:
        if isinstance(n_clusters, (bool, np.bool_)) or not isinstance(n_clusters, (int, np.integer)) or (not 1 <= n_clusters <= len(geometry.rho)):
            raise ValueError('n_clusters must be an integer in [1, minority training rows].')
        centers = np.argsort(-geometry.gamma, kind='stable')[:n_clusters]
    else:
        centers = center_indices
    return allocate_points(geometry, centers)
