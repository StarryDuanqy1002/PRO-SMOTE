import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y
from ..base import BaseSampler

def largest_remainder(total, masses):
    masses = np.asarray(masses, dtype=float)
    if total < 0 or int(total) != total or masses.ndim != 1:
        raise ValueError('Nonnegative integer total and 1D masses required.')
    if np.any(~np.isfinite(masses)) or np.any(masses < 0):
        raise ValueError('Masses must be finite and nonnegative.')
    if not len(masses):
        if total:
            raise ValueError('Cannot allocate a positive total to no clusters.')
        return np.empty(0, dtype=int)
    proportions = masses / masses.sum() if masses.sum() else np.full(len(masses), 1 / len(masses))
    raw = total * proportions
    counts = np.floor(raw).astype(int)
    remainder = int(total - counts.sum())
    counts[np.argsort(-(raw - counts), kind='stable')[:remainder]] += 1
    return counts

class RSMOTE(BaseSampler):

    def __init__(self, k=5, target_ratio=1.0, random_state=42):
        self.k = k
        self.target_ratio = target_ratio
        self.random_state = random_state

    def fit_resample(self, X, y):
        X, y = check_X_y(X, y, dtype=float)
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) != 2 or self.k < 1 or (not 0 < self.target_ratio <= 1):
            raise ValueError('Binary labels, k>=1 and target_ratio in (0,1] required.')
        minority = labels[np.argmin(counts)]
        ids = np.flatnonzero(y == minority)
        target = max(0, int(np.ceil(counts.max() * self.target_ratio)) - len(ids))
        self.n_requested_ = target
        self.n_generated_ = 0
        self.status_ = 'no_generation'
        if not target or len(ids) < 2:
            return (X.copy(), y.copy())
        k = min(self.k, len(X) - 1)
        neighbors = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X[ids], return_distance=False)
        neighbors = np.array([row[row != i][:k] for i, row in zip(ids, neighbors)])
        majority_count = (y[neighbors] != minority).sum(axis=1)
        kept = majority_count < k
        Xm = X[ids[kept]]
        m = majority_count[kept]
        if len(Xm) < 2:
            return (X.copy(), y.copy())
        kh = min(self.k, len(Xm) - 1)
        nh = NearestNeighbors(n_neighbors=kh + 1).fit(Xm).kneighbors(Xm, return_distance=False)
        nh = np.array([row[row != i][:kh] for i, row in enumerate(nh)])
        dh = np.linalg.norm(Xm[:, None, :] - Xm[nh], axis=2).sum(axis=1)
        Xmaj = X[y != minority]
        kt = min(self.k, len(Xmaj))
        dt = NearestNeighbors(n_neighbors=kt).fit(Xmaj).kneighbors(Xm)[0].sum(axis=1)
        rd = np.minimum(100, dt / np.maximum(dh, 1e-10))
        cluster = np.zeros(len(Xm), dtype=int)
        if len(Xm) >= 4 and np.ptp(rd) > 0:
            tentative = KMeans(n_clusters=2, n_init=10, random_state=self.random_state).fit_predict(rd[:, None])
            if np.bincount(tentative, minlength=2).min() >= 2:
                cluster = tentative
        groups = [np.flatnonzero(cluster == c) for c in np.unique(cluster)]
        quotas = largest_remainder(target, [len(g) for g in groups])
        rng = np.random.default_rng(self.random_state)
        generated = []
        for g, quota in zip(groups, quotas):
            pts = Xm[g]
            local_k = min(self.k, len(pts) - 1)
            near = NearestNeighbors(n_neighbors=local_k + 1).fit(pts).kneighbors(pts, return_distance=False)
            near = np.array([row[row != i][:local_k] for i, row in enumerate(near)])
            weights = (k - m[g]) / k + 1
            allocations = largest_remainder(int(quota), weights)
            for i, n in enumerate(allocations):
                for _ in range(n):
                    j = rng.choice(near[i])
                    generated.append(pts[i] + rng.random() * (pts[j] - pts[i]))
        samples = np.asarray(generated).reshape(-1, X.shape[1])
        self.n_generated_ = len(samples)
        self.status_ = 'complete'
        return (np.vstack([X, samples]), np.concatenate([y, np.full(len(samples), minority, dtype=y.dtype)]))
