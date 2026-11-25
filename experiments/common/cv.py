import hashlib
import importlib.metadata
import json
import platform
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits
from samplers.registry import make_sampler
from samplers.guarded import FIT_EVENTS
from .classifiers import make_classifier, classifier_specs
from datasets.splitting import outer_splits, inner_splitter
from datasets.splitting import exact_vector_groups, grouped_inner_splits
from datasets.preprocessing import build_pipeline
from .metrics import GMEAN_SCORER, METRICS, score_predictions
from utils.io import write_json, json_default
from datasets.loaders import ROOT, load_dataset
from utils.config import save_config
from utils.provenance import source_hashes, environment_identity

def compact_fit_events(events):
    fields = ('stage', 'inner_fold', 'input_sha256', 'class_counts', 'implementation', 'executed_implementation', 'effective_parameters', 'adjustments', 'status', 'requested', 'generated', 'shortfall', 'error')
    grouped = {}
    for event in events:
        details = {key: event[key] for key in fields if key in event}
        token = json.dumps(details, sort_keys=True, default=json_default)
        item = grouped.setdefault(token, dict(details, calls=0, cache_hits=0))
        item['calls'] += 1
        item['cache_hits'] += bool(event.get('served_from_cache'))
    return list(grouped.values())

def summarize(rows, expected_folds):
    frame = pd.DataFrame(rows)
    if frame.empty or 'f1' not in frame:
        return pd.DataFrame()
    ok = frame[frame['status'] == 'ok']
    keys = ['dataset', 'classifier', 'method']
    if ok.empty:
        return pd.DataFrame()
    available = [metric for metric in METRICS if metric in ok.columns]
    summary = ok.groupby(keys)[available].agg(['mean', 'std', 'count'])
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.reset_index()
    summary['complete'] = summary['f1_count'] == expected_folds
    return summary

def run_experiment(config, output, root=ROOT):
    from copy import deepcopy
    config = deepcopy(config)
    config.setdefault('classifier_specs', classifier_specs())
    if config.get('protocol_status') == 'revision_pro_j07_20261111':
        from experiments.revision import validate_config
        validate_config(config, subset=True, smoke=bool(config.get('gate_smoke')))
    output = Path(output)
    artifact_profile = config.get('artifact_profile', 'full')
    if artifact_profile not in ('full', 'compact'):
        raise ValueError('artifact_profile must be full or compact.')
    compact_artifacts = artifact_profile == 'compact'
    if config.get('seeds') and (not config.get('_seed_child')):
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Result directory is not empty: {output}')
        output.mkdir(parents=True, exist_ok=True)
        statuses = {}
        for value in config['seeds']:
            child = deepcopy(config)
            child.pop('seeds', None)
            child['seed'] = int(value)
            child['_seed_child'] = True
            statuses[str(value)] = run_experiment(child, output / f'seed_{value}', root=root)
        write_json(output / 'multi_seed_status.json', statuses)
        return dict(seeds=[int(x) for x in config['seeds']], statuses=statuses, complete=all((x['complete'] for x in statuses.values())))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Result directory is not empty: {output}. Choose a new output directory.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'splits').mkdir()
    (output / 'diagnostics').mkdir()
    (output / 'predictions').mkdir()
    (output / 'searches').mkdir()
    if not compact_artifacts:
        (output / 'sampling_events').mkdir()
    save_config(output / 'config.yaml', config)
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'scikit-learn', 'imbalanced-learn', 'pandas']}
    write_json(output / 'run.json', dict(config=config, python=platform.python_version(), dependencies=versions, source_sha256=source_hashes(), environment=environment_identity(), protocol='reconstruction; not recovered historical results'))
    rows, failures, dataset_meta = ([], [], [])
    folds, repeats = (config.get('folds', 5), config.get('repeats', 1))
    seed = config.get('seed', 42)
    for name in config['datasets']:
        try:
            X, y, meta = load_dataset(name, root)
            meta['sha256'] = hashlib.sha256((Path(root) / meta['path']).read_bytes()).hexdigest()
            dataset_meta.append(meta)
            if np.bincount(y).min() < folds:
                raise ValueError(f'{name}: fewer minority observations than outer folds.')
            policy = config.get('split_policy', 'stratified')
            if policy not in ('stratified', 'exact_vector_groups'):
                raise ValueError('Unknown split_policy.')
            groups = exact_vector_groups(X) if policy == 'exact_vector_groups' else None
            splits = outer_splits(X, y, folds, repeats, seed, groups=groups)
            if config.get('gate_outer_limit'):
                if not config.get('gate_smoke'):
                    raise ValueError('Outer-fold limit is permitted only for gate_smoke.')
                splits = splits[:int(config['gate_outer_limit'])]
            if groups is not None:
                write_json(output / 'splits' / f'{name}__groups.json', groups)
            np.savez_compressed(output / 'splits' / f'{name}.npz', **{f'{kind}_{i}': idx for i, pair in enumerate(splits) for kind, idx in zip(('train', 'test'), pair)})
        except (OSError, ValueError, KeyError) as error:
            failures.append(dict(dataset=name, stage='load', error=str(error)))
            continue
        for fold, (train, test) in enumerate(splits):
            for classifier_name in config['classifiers']:
                for method in config['methods']:
                    start = time.perf_counter()
                    record = dict(dataset=name, classifier=classifier_name, method=method, seed=seed, fold=fold, train_size=len(train), test_size=len(test))
                    caught = []
                    events = []
                    event_token = FIT_EVENTS.set(events)
                    inner_count = 0
                    search_finished = False
                    stem = f'{name}__{classifier_name}__{method}__{fold}'
                    try:
                        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                            warnings.simplefilter('always')
                            fold_seed = seed + fold
                            sampler = make_sampler(config.get('sampler_types', {}).get(method, method), fold_seed, config.get('sampler_parameters', {}).get(method, {}))
                            classifier, grid = make_classifier(classifier_name, fold_seed, config['classifier_specs'])
                            pipeline = build_pipeline(sampler, classifier)
                            if config.get('tune_classifiers', True):
                                inner = inner_splitter(y[train], config.get('inner_folds', 3), fold_seed)
                                if groups is None:
                                    inner = list(inner.split(X[train], y[train]))
                                else:
                                    inner, effective_inner_folds = grouped_inner_splits(X[train], y[train], groups[train], inner.n_splits, fold_seed)
                                    record['effective_inner_folds'] = effective_inner_folds
                                inner_count = len(inner)
                                write_json(output / 'searches' / f'{stem}__splits.json', [{'train': train[a], 'validation': train[b]} for a, b in inner])
                                search_grid = {f'classifier__{key}': value for key, value in grid.items()}
                                sampler_grid = config.get('search_grids', {}).get(method, {})
                                search_grid.update({key.replace('sampler__', 'sampler__estimator__', 1) if key.startswith('sampler__') and (not key.startswith('sampler__estimator__')) else key: value for key, value in sampler_grid.items()})
                                selection_metric = config.get('selection_metric', 'gmean')
                                scoring = GMEAN_SCORER if selection_metric == 'gmean' else selection_metric
                                search = GridSearchCV(pipeline, search_grid, scoring=scoring, cv=inner, n_jobs=1, error_score='raise', refit=True)
                                search.fit(X[train], y[train])
                                search_finished = True
                                if not compact_artifacts or config.get('main_review_evidence', False):
                                    pd.DataFrame(search.cv_results_).to_csv(output / 'searches' / f'{stem}.csv', index=False)
                                fitted = search.best_estimator_
                                record['best_parameters'] = json.dumps(search.best_params_, default=json_default)
                            else:
                                if config.get('search_grids'):
                                    raise ValueError('search_grids requires tune_classifiers=true.')
                                fitted = pipeline.fit(X[train], y[train])
                                record['best_parameters'] = '{}'
                            predicted = fitted.predict(X[test])
                            if hasattr(fitted, 'decision_function'):
                                scores = fitted.decision_function(X[test])
                            else:
                                scores = fitted.predict_proba(X[test])[:, 1]
                            record.update(score_predictions(y[test], predicted, scores), status='ok')
                            trained_sampler = fitted.named_steps['sampler']
                            audit = {'train_indices': train, 'test_indices': test, 'scaler_scale': fitted.named_steps['scaler'].scale_, 'scaler_mean': fitted.named_steps['scaler'].mean_, 'imputer_statistics': fitted.named_steps['imputer'].statistics_}
                            trained_classifier = fitted.named_steps['classifier']
                            audit['classifier_class'] = type(trained_classifier).__module__ + '.' + type(trained_classifier).__name__
                            audit['classifier_parameters'] = trained_classifier.get_params(deep=False)
                            if hasattr(trained_classifier, 'solver_backend_'):
                                audit['classifier_solver'] = {'backend': trained_classifier.solver_backend_, 'fit_status': int(trained_classifier.fit_status_), 'iterations': trained_classifier.n_iter_.tolist(), 'tol': trained_classifier.tol, 'kkt_gap': getattr(trained_classifier, 'solver_kkt_gap_', None)}
                            if trained_sampler != 'passthrough':
                                audit['sampler_parameters'] = trained_sampler.get_params()
                                audit['guard_policy'] = trained_sampler.audit_
                                for attr in ('n_requested_', 'n_generated_', 'shortfall_', 'status_'):
                                    record['sampler_' + attr.rstrip('_')] = getattr(trained_sampler, attr)
                                trained_sampler = trained_sampler.estimator_
                                if hasattr(trained_sampler, 'synthetic_samples_'):
                                    synthetic = np.asarray(trained_sampler.synthetic_samples_, dtype=np.float64)
                                    audit['synthetic_matrix'] = {'shape': list(synthetic.shape), 'dtype': str(synthetic.dtype), 'sha256': hashlib.sha256(synthetic.tobytes(order='C')).hexdigest()}
                                for attribute in ('cluster_diagnostics_', 'round_diagnostics_', 'feature_weights_', 'allocation_', 'effective_params_', 'status_', 'n_requested_', 'n_generated_', 'shortfall_'):
                                    if hasattr(trained_sampler, attribute):
                                        audit[attribute] = getattr(trained_sampler, attribute)
                            stem = f'{name}__{classifier_name}__{method}__{fold}'
                            write_json(output / 'diagnostics' / f'{stem}.json', audit)
                            pd.DataFrame(dict(row_index=test, truth=y[test], prediction=predicted, score=scores)).to_csv(output / 'predictions' / f'{stem}.csv', index=False)
                    except Exception as error:
                        record.update(status='failed', error=f'{type(error).__name__}: {error}')
                        failures.append(record.copy())
                    record['seconds'] = time.perf_counter() - start
                    FIT_EVENTS.reset(event_token)
                    for call, event in enumerate(events):
                        event.update(dataset=name, classifier=classifier_name, method=method, outer_fold=fold, seed=seed + fold, call=call)
                        if inner_count and (not (search_finished and call == len(events) - 1)):
                            event.update(stage='inner', candidate=call // inner_count, inner_fold=call % inner_count)
                        else:
                            event.update(stage='outer_refit')
                    record['sampler_fit_calls'] = len(events)
                    record['sampler_cache_hits'] = sum((bool(e.get('served_from_cache')) for e in events))
                    if compact_artifacts:
                        record['sampler_fit_summary'] = json.dumps(compact_fit_events(events), default=json_default)
                    if not compact_artifacts:
                        sampling_events_dir = output / 'sampling_events'
                        sampling_events_dir.mkdir(parents=True, exist_ok=True)
                        write_json(sampling_events_dir / f'{stem}.json', events)
                    record['warnings'] = json.dumps(sorted(set((str(w.message) for w in caught))))
                    rows.append(record)
                    pd.DataFrame(rows).to_csv(output / 'fold_metrics.csv', index=False)
                    print(f"{name} fold={fold + 1}/{len(splits)} {classifier_name} {method}: {record['status']} ({record['seconds']:.2f}s)", flush=True)
    write_json(output / 'datasets.json', dataset_meta)
    write_json(output / 'failures.json', failures)
    summary = summarize(rows, folds * repeats)
    summary.to_csv(output / 'summary.csv', index=False)
    expected_outer = min(folds * repeats, int(config.get('gate_outer_limit', folds * repeats)))
    expected = len(config['datasets']) * expected_outer * len(config['methods']) * len(config['classifiers'])
    status = dict(expected_fits=expected, successful_fits=sum((r['status'] == 'ok' for r in rows)), failed_fits=sum((r['status'] != 'ok' for r in rows)), load_failures=sum((f.get('stage') == 'load' for f in failures)), complete=not failures and len(rows) == expected)
    write_json(output / 'status.json', status)
    return status
