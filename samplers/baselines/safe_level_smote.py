import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y
from ..base import BaseSampler

class SafeLevelSMOTE(BaseSampler):
    neighbor_scope = 'all_training'
    proportion_rounding = 'floor'

    def __init__(self, proportion=1.0, n_neighbors=5, random_state=42):
        self.proportion = proportion
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        X, y = check_X_y(X, y, dtype=float)
        if not np.isfinite(self.proportion) or self.proportion < 0:
            raise ValueError('proportion must be finite and nonnegative')
        if not isinstance(self.n_neighbors, (int, np.integer)) or self.n_neighbors < 1:
            raise ValueError('n_neighbors must be a positive integer')
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) != 2:
            raise ValueError('SafeLevelSMOTE supports binary targets only')
        min_label = labels[np.argmin(counts)]
        n_to = max(0, int((counts.max() - counts.min()) * self.proportion))
        if n_to == 0 or counts.min() < 2:
            return (X.copy(), y.copy())
        k = min(self.n_neighbors + 1, len(X))
        indices = NearestNeighbors(n_neighbors=k, n_jobs=1).fit(X).kneighbors(X, return_distance=False)
        safe_levels = np.sum(y[indices[:, 1:]] == min_label, axis=1)
        minority = np.flatnonzero(y == min_label)
        positive = minority[safe_levels[minority] > 0]
        zero = minority[safe_levels[minority] == 0]
        part0 = np.vstack([np.repeat(positive, k - 1), indices[positive, 1:].flatten()]).T
        positions = np.where(safe_levels[indices[zero, 1:]] > 0)
        part1 = np.vstack([zero[positions[0]], indices[zero, 1:][positions]]).T
        pairs = np.vstack([part0, part1])
        if len(pairs) == 0:
            return (X.copy(), y.copy())
        levels = np.column_stack([pairs, safe_levels[pairs[:, 0]], safe_levels[pairs[:, 1]]])
        rng = check_random_state(self.random_state)
        chosen = rng.choice(np.arange(len(levels)), n_to)
        gap = rng.random_sample(size=(n_to, X.shape[1]))
        slp, sln = (levels[chosen, 2], levels[chosen, 3])
        gap = (gap.T * np.where((sln == 0) & (slp > 0), 0.0, 1.0)).T
        mask = slp > sln
        gap[mask] = (gap[mask].T * sln[mask] / slp[mask]).T
        mask = slp < sln
        gap[mask] = (gap[mask].T * slp[mask] / sln[mask]).T
        gap[mask] = (gap[mask].T + (1.0 - slp[mask] / sln[mask])).T
        samples = X[levels[chosen, 0]] + (X[levels[chosen, 1]] - X[levels[chosen, 0]]) * gap
        return (np.vstack([X, samples]), np.hstack([y, np.repeat(min_label, len(samples))]))
