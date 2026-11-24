"""PRO-SMOTE oversampler for binary classification.

Call ``PROSMOTE(...).fit_resample(X, y)``. Requires NumPy, SciPy, and scikit-learn.
"""
from collections import deque
import math
import warnings

import numpy as np
from scipy.spatial.distance import jensenshannon, pdist, squareform
from scipy.stats import hypergeom
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y


def _probabilities(values):
    values = np.asarray(values, dtype=float)
    total = values.sum()
    probabilities = values / total if total else np.full(len(values), 1 / len(values))
    probabilities = np.maximum(probabilities, np.finfo(float).eps)
    return probabilities / probabilities.sum()


def _js_bits(p, q):
    return float(jensenshannon(p, q, base=2.0) ** 2)


def _kulczynski(y_true, y_pred):
    true_positive = np.count_nonzero(y_true & y_pred)
    false_positive = np.count_nonzero(~y_true & y_pred)
    false_negative = np.count_nonzero(y_true & ~y_pred)
    if true_positive + false_positive == 0 or true_positive + false_negative == 0:
        return 0.0
    return 0.5 * (
        true_positive / (true_positive + false_positive)
        + true_positive / (true_positive + false_negative)
    )


def _feature_weights(X, minority, random_state):
    scores = np.zeros(X.shape[1])
    for feature in range(X.shape[1]):
        column = X[:, feature]
        if np.ptp(column) == 0:
            continue
        bins = KMeans(n_clusters=2, n_init=10, random_state=random_state).fit_predict(
            column.reshape(-1, 1)
        )
        scores[feature] = max(
            _kulczynski(minority, bins == 0),
            _kulczynski(minority, bins == 1),
        )
    total = scores.sum()
    weights = scores / total if total else np.full(len(scores), 1 / len(scores))
    return weights


def _largest_remainder(total, masses):
    masses = np.asarray(masses, dtype=float)
    if total < 0 or int(total) != total or masses.ndim != 1:
        raise ValueError("A nonnegative integer total and one-dimensional masses are required.")
    if np.any(~np.isfinite(masses)) or np.any(masses < 0):
        raise ValueError("Allocation masses must be finite and nonnegative.")
    if not len(masses):
        if total:
            raise ValueError("Cannot allocate samples without eligible clusters.")
        return np.empty(0, dtype=int)
    mass_sum = masses.sum()
    proportions = masses / mass_sum if mass_sum else np.full(len(masses), 1 / len(masses))
    raw = total * proportions
    allocation = np.floor(raw).astype(int)
    remainder = int(total - allocation.sum())
    allocation[np.argsort(-(raw - allocation), kind="stable")[:remainder]] += 1
    return allocation


def _simplex_weights(random_state, count):
    coefficients = random_state.uniform(size=(count, 2))
    reflect = coefficients.sum(axis=1) > 1
    coefficients[reflect] = 1 - coefficients[reflect]
    return np.column_stack((coefficients, 1 - coefficients.sum(axis=1)))


def _snn_dpc(X, k, eps, min_samples, zero_cluster_count, memory_limit_mb):
    n_samples = len(X)
    estimated_mb = 32 * n_samples * n_samples / 1024**2
    if estimated_mb > memory_limit_mb:
        raise MemoryError(
            f"SNN-DPC requires approximately {estimated_mb:.0f} MiB of pairwise "
            f"working memory; limit is {memory_limit_mb} MiB."
        )

    dbscan_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    dbscan_count = len(set(dbscan_labels) - {-1})
    n_centers = min(n_samples, dbscan_count or zero_cluster_count)
    k = min(k, n_samples)

    distances = squareform(pdist(X))
    np.fill_diagonal(distances, -1.0)
    order = np.argsort(distances, axis=1, kind="stable")
    np.fill_diagonal(distances, 0.0)
    neighbors = order[:, :k]
    neighbor_sets = [set(row.tolist()) for row in neighbors]

    similarities = [[] for _ in range(n_samples)]
    shared_edges = {}
    for i in range(n_samples):
        for j in neighbors[i]:
            j = int(j)
            if j == i:
                continue
            shared = neighbor_sets[i] & neighbor_sets[j]
            shared_edges[i, j] = len(shared)
            if j < i and i in neighbor_sets[j]:
                indices = np.fromiter(sorted(shared), dtype=int)
                denominator = float((distances[i, indices] + distances[j, indices]).sum())
                similarity = len(shared) ** 2 / denominator if denominator > 1e-9 else 0.0
                similarities[i].append(similarity)
                similarities[j].append(similarity)

    density = np.array([sum(sorted(row)[-k:]) for row in similarities])
    density_order = np.argsort(-density, kind="stable")
    neighbor_distance_sums = distances[np.arange(n_samples)[:, None], neighbors].sum(axis=1)
    delta = np.zeros(n_samples)
    for rank, sample in enumerate(density_order[1:], 1):
        earlier = density_order[:rank]
        delta[sample] = np.min(
            distances[sample, earlier]
            * (neighbor_distance_sums[sample] + neighbor_distance_sums[earlier])
        )
    delta[density_order[0]] = delta.max()
    centers = np.sort(np.argsort(-(density * delta), kind="stable")[:n_centers])

    if n_centers == 1:
        return np.zeros(n_samples, dtype=int), centers

    labels = np.full(n_samples, -1, dtype=int)
    labels[centers] = np.arange(n_centers)
    queue = deque(centers.tolist())
    while queue:
        sample = queue.popleft()
        for neighbor in neighbors[sample]:
            neighbor = int(neighbor)
            if labels[neighbor] < 0 and shared_edges.get((sample, neighbor), 0) >= k / 2:
                labels[neighbor] = labels[sample]
                queue.append(neighbor)

    width = k
    while np.any(labels < 0):
        remaining = np.flatnonzero(labels < 0)
        votes = np.zeros((len(remaining), n_centers), dtype=int)
        for start in range(0, len(remaining), 256):
            stop = min(start + 256, len(remaining))
            known = labels[order[remaining[start:stop], :width]]
            rows, columns = np.nonzero(known >= 0)
            votes[start:stop] = np.bincount(
                rows * n_centers + known[rows, columns],
                minlength=(stop - start) * n_centers,
            ).reshape(stop - start, n_centers)
        best = votes.max()
        if best:
            selected = votes.max(axis=1) == best
            labels[remaining[selected]] = votes[selected].argmax(axis=1)
        elif width < n_samples:
            first_known = n_samples - 1
            for start in range(0, len(remaining), 256):
                rows = remaining[start:start + 256]
                first_known = min(
                    first_known,
                    int((labels[order[rows]] >= 0).argmax(axis=1).min()),
                )
            width = max(width + 1, first_known + 1)
        else:
            labels[remaining] = distances[np.ix_(remaining, centers)].argmin(axis=1)
    return labels, centers


def _local_density(X, feature_weights, k):
    distances = squareform(pdist(X * np.sqrt(feature_weights)))
    np.fill_diagonal(distances, np.inf)
    k_effective = min(k, len(X) - 1)
    mean_distance = np.sort(distances, axis=1)[:, :k_effective].mean(axis=1)
    return np.exp(-mean_distance), distances


class PROSMOTE(BaseEstimator):
    """Structure-aware oversampling for binary classification."""

    def __init__(
        self,
        k_neighbors=5,
        target_ratio=1.0,
        dbscan_eps=0.2,
        dbscan_min_samples=3,
        js_lower_tail=0.1,
        lof_neighbors=30,
        lof_contamination=0.1,
        max_compensation_rounds=12,
        zero_cluster_count=1,
        memory_limit_mb=1024,
        random_state=42,
    ):
        self.k_neighbors = k_neighbors
        self.target_ratio = target_ratio
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.js_lower_tail = js_lower_tail
        self.lof_neighbors = lof_neighbors
        self.lof_contamination = lof_contamination
        self.max_compensation_rounds = max_compensation_rounds
        self.zero_cluster_count = zero_cluster_count
        self.memory_limit_mb = memory_limit_mb
        self.random_state = random_state

    def _validate_parameters(self):
        for name in (
            "k_neighbors",
            "dbscan_min_samples",
            "lof_neighbors",
            "max_compensation_rounds",
            "zero_cluster_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("dbscan_eps", "memory_limit_mb"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not np.isfinite(self.target_ratio) or not 0 < self.target_ratio <= 1:
            raise ValueError("target_ratio must be in (0, 1].")
        if not np.isfinite(self.js_lower_tail) or not 0 < self.js_lower_tail < 1:
            raise ValueError("js_lower_tail must be strictly between zero and one.")
        if self.lof_contamination != "auto" and (
            not np.isfinite(self.lof_contamination)
            or not 0 < self.lof_contamination <= 0.5
        ):
            raise ValueError("lof_contamination must be 'auto' or in (0, 0.5].")

    def _eligible_clusters(self, labels, minority_mask):
        population = len(minority_mask)
        minority_count = int(minority_mask.sum())
        eligible = []
        for cluster in np.unique(labels):
            mask = labels == cluster
            size = int(mask.sum())
            count = int(minority_mask[mask].sum())
            proportion = count / size
            lower_count = int(hypergeom.ppf(
                self.js_lower_tail, population, minority_count, size
            ))
            if hypergeom.cdf(lower_count, population, minority_count, size) <= self.js_lower_tail:
                lower_count += 1
            cutoff = _js_bits(
                [lower_count / size, 1 - lower_count / size], [1.0, 0.0]
            )
            divergence = _js_bits([proportion, 1 - proportion], [1.0, 0.0])
            # J07 permits two-point interpolation only when both vertices are available.
            if count >= 2 and divergence <= cutoff + 1e-14:
                eligible.append(int(cluster))
        return eligible

    def _build_pools(self, X, labels, minority_mask, feature_weights, clusters):
        pools = []
        masses = []
        for cluster in clusters:
            points = X[(labels == cluster) & minority_mask]
            density, distances = _local_density(points, feature_weights, self.k_neighbors)
            neighbors = np.argsort(distances, axis=1, kind="stable")[:, :-1]
            pools.append((points, _probabilities(density), density, neighbors))
            masses.append(float(density.sum()))
        return pools, masses

    def _generate(self, pools, allocation):
        samples = []
        for (points, anchor_probabilities, density, neighbors), count in zip(pools, allocation):
            if count == 0:
                continue
            anchors = self.random_state_.choice(len(points), count, p=anchor_probabilities)
            if len(points) == 2:
                partners = np.empty(count, dtype=int)
                for i, anchor in enumerate(anchors):
                    allowed = neighbors[anchor]
                    partners[i] = self.random_state_.choice(
                        allowed, 1, replace=False, p=_probabilities(density[allowed])
                    )[0]
                coefficient = self.random_state_.uniform(size=(count, 1))
                samples.append(
                    (1 - coefficient) * points[anchors]
                    + coefficient * points[partners]
                )
                continue

            pairs = np.empty((count, 2), dtype=int)
            for i, anchor in enumerate(anchors):
                allowed = neighbors[anchor]
                pairs[i] = self.random_state_.choice(
                    allowed, 2, replace=False, p=_probabilities(density[allowed])
                )
            if np.any(pairs[:, 0] == pairs[:, 1]) or np.any(pairs == anchors[:, None]):
                raise AssertionError("Three distinct source samples are required.")
            coefficients = _simplex_weights(self.random_state_, count)
            samples.append(
                coefficients[:, :1] * points[anchors]
                + coefficients[:, 1:2] * points[pairs[:, 0]]
                + coefficients[:, 2:] * points[pairs[:, 1]]
            )
        return np.vstack(samples) if samples else np.empty((0, self.n_features_in_))

    def _filter(self, X, candidates, feature_weights):
        combined = np.vstack((X, candidates)) * np.sqrt(feature_weights)
        n_neighbors = min(self.lof_neighbors, len(combined) - 1)
        labels = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.lof_contamination,
        ).fit_predict(combined)
        return candidates[labels[len(X):] == 1]

    def fit_resample(self, X, y):
        self._validate_parameters()
        X, y = check_X_y(X, y, dtype=float, copy=True)
        self.n_features_in_ = X.shape[1]
        self.random_state_ = check_random_state(self.random_state)

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("PRO-SMOTE supports binary labels only.")
        minority_class = classes[np.argmin(counts)]
        minority_mask = y == minority_class
        self.minority_class_ = minority_class
        self.n_requested_ = max(
            0,
            math.ceil((~minority_mask).sum() * self.target_ratio - minority_mask.sum()),
        )
        self.n_generated_ = 0
        self.shortfall_ = self.n_requested_
        if self.n_requested_ == 0:
            return X, y

        cluster_labels, _ = _snn_dpc(
            X,
            self.k_neighbors,
            self.dbscan_eps,
            self.dbscan_min_samples,
            self.zero_cluster_count,
            self.memory_limit_mb,
        )
        clusters = self._eligible_clusters(cluster_labels, minority_mask)
        if not clusters:
            warnings.warn(
                "No eligible clusters; returning the original data.",
                RuntimeWarning,
                stacklevel=2,
            )
            return X, y

        feature_weights = _feature_weights(X, minority_mask, self.random_state_)
        pools, masses = self._build_pools(
            X, cluster_labels, minority_mask, feature_weights, clusters
        )

        accepted = np.empty((0, X.shape[1]))
        for _ in range(self.max_compensation_rounds):
            missing = self.n_requested_ - len(accepted)
            candidates = self._generate(pools, _largest_remainder(missing, masses))
            accepted = np.vstack((accepted, self._filter(X, candidates, feature_weights)))
            if len(accepted) >= self.n_requested_:
                break
        if len(accepted) > self.n_requested_:
            accepted = accepted[
                self.random_state_.choice(len(accepted), self.n_requested_, replace=False)
            ]

        self.n_generated_ = len(accepted)
        self.shortfall_ = self.n_requested_ - self.n_generated_
        if self.shortfall_:
            warnings.warn(
                f"Generated {self.n_generated_} of {self.n_requested_} requested "
                "samples after bounded compensation.",
                RuntimeWarning,
                stacklevel=2,
            )
        X_resampled = np.vstack((X, accepted))
        y_resampled = np.concatenate(
            (y, np.full(len(accepted), self.minority_class_, dtype=y.dtype))
        )
        return X_resampled, y_resampled
