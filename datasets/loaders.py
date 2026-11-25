import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]

def load_dataset(name, root=ROOT):
    root = Path(root)
    index = json.loads((root / 'datasets' / 'index.json').read_text(encoding='utf-8'))
    spec = index[name.lower()]
    path = root / spec['path']
    skiprows = 0
    if path.suffix.lower() in ('.dat', '.arff'):
        lines = path.read_text(encoding='utf-8-sig').splitlines()
        locations = [i for i, line in enumerate(lines) if line.strip().lower() == '@data']
        if not locations:
            raise ValueError(f'No @data marker: {path}')
        skiprows = locations[0] + 1
    frame = pd.read_csv(path, header=spec.get('header'), skiprows=skiprows, skipinitialspace=True, na_values=['?', 'NaN', 'nan'], dtype=str)
    target_index = spec.get('target_column', -1) % frame.shape[1]
    target = frame.iloc[:, target_index]
    if target.isna().any():
        raise ValueError(f'{name}: missing target labels; specify a cleaning decision explicitly.')
    target = target.str.strip().str.strip('\'"')
    labels, counts = np.unique(target, return_counts=True)
    if len(labels) != 2:
        raise ValueError(f'{name}: expected binary target, got {labels.tolist()}; check header configuration.')
    minority_label = labels[np.argmin(counts)]
    y = (target.to_numpy() == minority_label).astype(int)
    features = frame.drop(columns=frame.columns[target_index])
    X = features.apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    if np.isinf(X).any():
        raise ValueError(f'{name}: infinite feature values.')
    metadata = dict(name=name, path=spec['path'], rows=len(y), features=X.shape[1], minority=int(y.sum()), majority=int((y == 0).sum()), minority_label=str(minority_label), missing_values=int(np.isnan(X).sum()), missing_rows=int(np.isnan(X).any(axis=1).sum()))
    return (X, y, metadata)
