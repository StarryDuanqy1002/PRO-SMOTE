import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedGroupKFold

def outer_splits(X, y, folds=5, repeats=1, seed=42, groups=None):
    if np.unique(y, return_counts=True)[1].min() < folds:
        raise ValueError('Fewer minority observations than outer folds.')
    if groups is not None:
        result = []
        for repeat in range(repeats):
            cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
            parts = list(cv.split(X, y, groups))
            validate_group_splits(y, groups, parts)
            result.extend(parts)
        return result
    if repeats == 1:
        return list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y))
    return list(RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed).split(X, y))

def inner_splitter(y_train, folds=3, seed=42):
    count = min(folds, int(np.unique(y_train, return_counts=True)[1].min()))
    if count < 2:
        raise ValueError('Insufficient minority samples for inner CV.')
    return StratifiedKFold(n_splits=count, shuffle=True, random_state=seed)

def grouped_inner_splits(X, y, groups, folds=3, seed=42):
    maximum = min(folds, int(np.unique(y, return_counts=True)[1].min()))
    for count in range(maximum, 1, -1):
        cv = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
        parts = list(cv.split(X, y, groups))
        try:
            validate_group_splits(y, groups, parts)
        except ValueError:
            continue
        return (parts, count)
    raise ValueError('No valid grouped inner split with both classes; outer method-fold fails explicitly.')

def exact_vector_groups(X):
    import hashlib
    values = np.asarray(X, dtype=np.float64).copy()
    values[values == 0] = 0
    values[np.isnan(values)] = np.nan
    keys = [hashlib.sha256(row.tobytes()).hexdigest() for row in values]
    return np.unique(keys, return_inverse=True)[1]

def validate_group_splits(y, groups, splits):
    for a, b in splits:
        if set(groups[a]) & set(groups[b]):
            raise ValueError('Group overlap across split.')
        if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
            raise ValueError('Group split missing a class; not silently scored.')
