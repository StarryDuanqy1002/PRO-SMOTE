from sklearn.base import BaseEstimator

class BaseSampler(BaseEstimator):
    _estimator_type = 'sampler'

    def fit_resample(self, X, y):
        raise NotImplementedError
