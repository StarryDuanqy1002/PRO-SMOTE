import numpy as np
from sklearn.metrics import pairwise_distances_chunked
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y
from ..base import BaseSampler

class SYMPROD(BaseSampler):
    proportion_rounding = 'floor'

    def __init__(self, proportion=1.0, std_outliers=3, k_neighbors=7, m_neighbors=7, cutoff_threshold=1.25, random_state=42):
        self.proportion = proportion
        self.std_outliers = std_outliers
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.cutoff_threshold = cutoff_threshold
        self.random_state = random_state

    @staticmethod
    def _closeness(points):
        sums = np.concatenate(list(pairwise_distances_chunked(points, working_memory=64, reduce_func=lambda d, start: d.sum(axis=1))))
        cf = 1.0 / np.maximum(sums, 1e-12)
        span = np.ptp(cf)
        return np.zeros(len(cf)) if span <= 1e-12 else (cf - cf.min()) / span

    @staticmethod
    def _selection(Xm, Xj, k, cutoff):
        cf_m, cf_j = (SYMPROD._closeness(Xm), SYMPROD._closeness(Xj))
        d_mm, i_mm = NearestNeighbors(n_neighbors=k).fit(Xm).kneighbors()
        d_mj, i_mj = NearestNeighbors(n_neighbors=k).fit(Xj).kneighbors(Xm)
        tau_m = (cf_m[i_mm] / np.maximum(d_mm, 1e-12)).sum(axis=1)
        tau_j = (cf_j[i_mj] / np.maximum(d_mj, 1e-12)).sum(axis=1)
        mask = tau_m >= cutoff * tau_j
        phi = (tau_m[mask] + 1.0) / (tau_j[mask] + 1.0)
        probability = phi / phi.sum() if len(phi) else phi
        return (mask, probability)

    @staticmethod
    def _convex_coefficients(beta, probability):
        weights = beta * probability
        return weights / weights.sum()

    def fit_resample(self, X, y):
        X, y = check_X_y(X, y, dtype=float)
        if not np.isfinite(self.proportion) or self.proportion < 0:
            raise ValueError('proportion must be finite and nonnegative')
        if not np.isfinite(self.std_outliers) or not np.isfinite(self.cutoff_threshold) or self.std_outliers <= 0 or (self.cutoff_threshold < 0):
            raise ValueError('invalid SyMProD screening thresholds')
        if any((int(v) != v or v < 1 for v in (self.k_neighbors, self.m_neighbors))):
            raise ValueError('neighbor counts must be positive integers')
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) != 2:
            raise ValueError('SYMPROD supports binary targets only')
        min_label, maj_label = (labels[np.argmin(counts)], labels[np.argmax(counts)])
        n_to = int((counts.max() - counts.min()) * self.proportion)
        self.diagnostics_ = dict(requested=n_to, implementation='paper_equations_2_to_8', cutoff=float(self.cutoff_threshold))
        if n_to == 0 or counts.min() < 2:
            self.diagnostics_['reason'] = 'target_met_or_too_few_minority'
            return (X.copy(), y.copy())
        Xm_raw, Xj_raw = (X[y == min_label], X[y == maj_label])

        def screen(points):
            z = StandardScaler().fit_transform(points)
            return points[np.max(np.abs(z), axis=1) < self.std_outliers]
        Xm, Xj = (screen(Xm_raw), screen(Xj_raw))
        self.diagnostics_.update(minority_after_noise=len(Xm), majority_after_noise=len(Xj))
        if len(Xm) < 2 or len(Xj) < 1:
            self.diagnostics_['reason'] = 'insufficient_noise_filtered_candidates'
            return (X.copy(), y.copy())
        scaler = StandardScaler().fit(np.vstack([Xm, Xj]))
        Xm, Xj = (scaler.transform(Xm), scaler.transform(Xj))
        k = min(int(self.k_neighbors), len(Xm) - 1, len(Xj))
        mask, probability = self._selection(Xm, Xj, k, self.cutoff_threshold)
        anchors = Xm[mask]
        self.diagnostics_.update(effective_k=k, usable_anchors=len(anchors))
        if len(anchors) < 2:
            self.diagnostics_['reason'] = 'insufficient_screened_anchors'
            return (X.copy(), y.copy())
        m = min(int(self.m_neighbors), len(anchors) - 1)
        neighbors = NearestNeighbors(n_neighbors=m).fit(anchors).kneighbors(return_distance=False)
        self.diagnostics_.update(effective_m=m, reason='generated')
        rng = np.random.default_rng(self.random_state)
        samples = np.empty((n_to, X.shape[1]))
        for row in range(n_to):
            anchor = rng.choice(len(anchors), p=probability)
            pool = np.concatenate(([anchor], neighbors[anchor]))
            beta = np.maximum(rng.random(len(pool)), np.finfo(float).tiny)
            weights = self._convex_coefficients(beta, probability[pool])
            samples[row] = weights @ anchors[pool]
        return (np.vstack([X, scaler.inverse_transform(samples)]), np.concatenate([y, np.full(n_to, min_label, dtype=y.dtype)]))
