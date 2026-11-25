from imblearn.over_sampling import SMOTE
from .baselines.kmeans_smote import KMeansSMOTE
from .baselines.rsmote import RSMOTE
from .baselines.safe_level_smote import SafeLevelSMOTE
from .baselines.sd_csmote import SDC_SMOTE
from .baselines.smote_rsb import SMOTERSB
from .baselines.symprod import SYMPROD

def make_sampler(name, random_state=42, **parameters):
    samplers = {'SMOTE': SMOTE, 'KMeansSMOTE': KMeansSMOTE, 'RSMOTE': RSMOTE, 'Safe-Level-SMOTE': SafeLevelSMOTE, 'SMOTE-RSB': SMOTERSB, 'SD-CSMOTE': SDC_SMOTE, 'SyMProD': SYMPROD}
    try:
        sampler = samplers[name]
    except KeyError as error:
        raise ValueError(f'Unknown baseline: {name}') from error
    return sampler(random_state=random_state, **parameters)
