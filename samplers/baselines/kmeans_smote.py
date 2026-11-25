from imblearn.over_sampling import KMeansSMOTE as LibraryKMeansSMOTE
from imblearn.over_sampling import SMOTE

class KMeansSMOTE(LibraryKMeansSMOTE):

    def _fit_resample(self, X, y):
        self.no_eligible_clusters_ = False
        self.fallback_method_ = None
        try:
            return super()._fit_resample(X, y)
        except RuntimeError as error:
            if not str(error).startswith('No clusters found with sufficient samples of class '):
                raise
            self.no_eligible_clusters_ = True
            self.fallback_method_ = 'SMOTE'
            targets = {label: int((y == label).sum()) + int(amount) for label, amount in self.sampling_strategy_.items()}
            self.fallback_sampler_ = SMOTE(sampling_strategy=targets, random_state=self.random_state, k_neighbors=self.k_neighbors)
            return self.fallback_sampler_.fit_resample(X, y)
