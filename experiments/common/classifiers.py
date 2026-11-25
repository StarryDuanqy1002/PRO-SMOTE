from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from .stable_svc import StableSVC
from utils.config import load_config
from utils.paths import ROOT

def classifier_specs():
    return load_config(ROOT / 'configs/models/classifiers.yaml')

def make_classifier(name, seed, specs=None):
    registry = {'SVM': StableSVC, 'KNN': KNeighborsClassifier, 'AdaBoost': AdaBoostClassifier}
    if name not in registry:
        raise ValueError(f'Unknown classifier: {name}')
    spec = (specs if specs is not None else classifier_specs())[name]
    params = dict(spec.get('parameters', {}))
    if name == 'SVM':
        params.setdefault('shrinking', False)
    if name != 'KNN':
        params.setdefault('random_state', seed)
    return (registry[name](**params), spec.get('grid', {}))
