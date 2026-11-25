import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y
from ..base import BaseSampler

class SMOTERSB(BaseSampler):
    proportion_rounding = 'floor'

    def __init__(self, proportion=1.0, n_neighbors=5, random_state=42):
        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def _candidates(self, minority, count):
        k = min(len(minority) - 1, int(self.n_neighbors))
        indices = NearestNeighbors(n_neighbors=k, n_jobs=1).fit(minority).kneighbors(return_distance=False)
        pairs = np.column_stack((np.repeat(np.arange(len(minority)), k), indices.reshape(-1)))
        rng = check_random_state(self.random_state)
        selected = pairs[rng.randint(len(pairs), size=count)]
        gap = rng.random_sample((count, 1))
        return minority[selected[:, 0]] + gap * (minority[selected[:, 1]] - minority[selected[:, 0]])

    @staticmethod
    def _filter(candidates, majority, original, imbalance):
        scale = np.ptp(original, axis=0)
        scale[scale == 0] = 1.0
        similarity = 1.0 - pairwise_distances(candidates / scale, majority / scale, metric='minkowski', p=1) / candidates.shape[1]
        threshold = 0.4
        selected = np.zeros((0, candidates.shape[1]))
        while len(selected) == 0 and threshold <= 0.9:
            mask = np.sum(similarity > threshold, axis=1) == 0
            selected = candidates[mask]
            threshold += 0.05
        if len(selected) == 0:
            selected = candidates
        return selected[:min(len(selected), imbalance)]

    def fit_resample(self, X, y):
        X, y = check_X_y(X, y, dtype=float)
        if not np.isfinite(self.proportion) or self.proportion < 0:
            raise ValueError('proportion must be finite and nonnegative')
        if int(self.n_neighbors) != self.n_neighbors or self.n_neighbors < 1:
            raise ValueError('n_neighbors must be a positive integer')
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) != 2:
            raise ValueError('SMOTERSB supports binary targets only')
        minority_label, majority_label = (labels[np.argmin(counts)], labels[np.argmax(counts)])
        imbalance = int(counts.max() - counts.min())
        self.n_requested_ = int(imbalance * self.proportion)
        if self.n_requested_ == 0 or counts.min() < 2:
            self.n_generated_, self.shortfall_ = (0, self.n_requested_)
            return (X.copy(), y.copy())
        candidates = self._candidates(X[y == minority_label], self.n_requested_)
        samples = self._filter(candidates, X[y == majority_label], X, imbalance)
        self.n_generated_ = len(samples)
        self.shortfall_ = max(0, self.n_requested_ - self.n_generated_)
        return (np.vstack([X, samples]), np.concatenate([y, np.full(len(samples), minority_label, dtype=y.dtype)]))
