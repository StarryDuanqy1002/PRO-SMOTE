import numpy as np
from scipy.special import logsumexp
from sklearn.utils.validation import check_X_y
from ..base import BaseSampler
from ._sd_csmote_snn import snn_dpc

class SDC_SMOTE(BaseSampler):
    version = 'independent-snn-fixed-count-20261110'
    neighbor_scope = 'minority_including_self'

    def __init__(self, proportion=1.0, k_neighbors=5, n_clusters=2, center_indices=None, memory_limit_mb=1024, random_state=42):
        self.proportion = proportion
        self.k_neighbors = k_neighbors
        self.n_clusters = n_clusters
        self.center_indices = center_indices
        self.memory_limit_mb = memory_limit_mb
        self.random_state = random_state

    @staticmethod
    def _cluster_log_density(points, centre):
        n, d = points.shape
        sigma = float(np.std(points, axis=0, ddof=1).mean()) if n > 1 else 0.0
        h = max((4.0 / (3.0 * n)) ** 0.2 * sigma, 1e-06)
        z = (points - centre) / h
        log_density = logsumexp(-0.5 * np.sum(z * z, axis=1)) - np.log(n) - d * np.log(h) - 0.5 * d * np.log(2.0 * np.pi)
        return (float(log_density), h)

    @staticmethod
    def _allocate(total, masses):
        raw = total * masses / masses.sum()
        counts = np.floor(raw).astype(int)
        remainder = int(total - counts.sum())
        counts[np.argsort(-(raw - counts), kind='stable')[:remainder]] += 1
        return counts

    def fit_resample(self, X, y):
        X, y = check_X_y(X, y, dtype=float)
        if not np.isfinite(self.proportion) or self.proportion < 0:
            raise ValueError('proportion must be finite and nonnegative.')
        for name in ('k_neighbors', 'n_clusters'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if not np.isfinite(self.memory_limit_mb) or self.memory_limit_mb <= 0:
            raise ValueError('memory_limit_mb must be finite and positive.')
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) != 2:
            raise ValueError('SDC_SMOTE supports binary targets only.')
        min_label = labels[np.argmin(counts)]
        self.n_features_in_ = X.shape[1]
        self.n_requested_ = int(np.ceil((counts.max() - counts.min()) * self.proportion))
        self.n_generated_ = 0
        self.status_ = 'no_generation'
        self.shortfall_ = self.n_requested_
        if self.n_requested_ == 0 or counts.min() < 2:
            return (X.copy(), y.copy())
        X_min = X[y == min_label]
        self.effective_k_ = min(self.k_neighbors, len(X_min))
        self.effective_n_clusters_ = min(self.n_clusters, len(X_min))
        result = snn_dpc(X_min, k=self.effective_k_, n_clusters=self.effective_n_clusters_, center_indices=self.center_indices, memory_limit_mb=self.memory_limit_mb)
        self.clustering_ = result
        self.cluster_labels_ = result.labels
        self.center_indices_ = result.centers
        self.cluster_centers_ = X_min[result.centers].copy()
        self.effective_n_clusters_ = len(result.centers)
        pools = [X_min[result.labels == cid] for cid in range(len(result.centers))]
        density = [self._cluster_log_density(points, centre) for points, centre in zip(pools, self.cluster_centers_)]
        self.cluster_log_densities_ = np.array([row[0] for row in density])
        self.cluster_bandwidths_ = np.array([row[1] for row in density])
        log_sparsity = -self.cluster_log_densities_
        masses = np.exp(log_sparsity - log_sparsity.max())
        self.cluster_weights_ = masses / masses.sum()
        self.allocation_ = self._allocate(self.n_requested_, masses)
        self.cluster_sizes_ = np.array([len(points) for points in pools])
        rng = np.random.default_rng(self.random_state)
        generated = []
        for points, centre, quota in zip(pools, self.cluster_centers_, self.allocation_):
            for _ in range(int(quota)):
                ids = rng.choice(len(points), 2, replace=len(points) == 1)
                generated.append((centre + points[ids[0]] + points[ids[1]]) / 3.0)
        samples = np.asarray(generated).reshape(-1, X.shape[1])
        self.n_generated_ = len(samples)
        self.shortfall_ = self.n_requested_ - self.n_generated_
        self.status_ = 'complete' if not self.shortfall_ else 'shortfall'
        return (np.vstack((X, samples)), np.concatenate((y, np.full(len(samples), min_label, dtype=y.dtype))))
