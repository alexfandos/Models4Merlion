from merlion.models.anomaly.base import DetectorConfig
from merlion.post_process.threshold import Threshold
from merlion.evaluate.anomaly import TSADMetric
from merlion.models.anomaly.base import DetectorBase
from merlion.transform.resample import TemporalResample


import pandas as pd
import numpy as np
import math
from scipy.stats import norm
from scipy.stats import entropy


class AAMPConfig(DetectorConfig):
 
    _default_transform = TemporalResample(granularity=None)

    _default_threshold = Threshold(alm_threshold=5)

    def __init__(self, segment_length = 290, minimal_compare_distance = 290, **kwargs):
        
        self.segment_length = segment_length
        self.minimal_compare_distance = minimal_compare_distance
        super().__init__(**kwargs)

class AAMP(DetectorBase):

    config_class = AAMPConfig

    # By default, we would like to train the model's post-rule (i.e. the threshold
    # at which we fire an alert) to maximize F1 score
    _default_post_rule_train_config = dict(metric=TSADMetric.F1)

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return True

    def __init__(self,  config: AAMPConfig):

        super().__init__(config)



    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:
        return self.process(train_data)

        

    def _get_anomaly_score(self, time_series: pd.DataFrame, time_series_prev: pd.DataFrame = None) -> pd.DataFrame:
        return self.process(time_series)

    
    def process(self, time_series):

        T = time_series.to_numpy()
        l = self.config.segment_length
        minDistance = self.config.minimal_compare_distance

        P = np.ones(len(T) - l + 1) * float('inf')

        for k in range(minDistance, len(T) - l + 1):
            i = 0
            j = k
            dist = self.compare(self.subsequence(T, i, l), self.subsequence(T, j, l))


            if P[i] > dist:
                P[i] = dist
            if P[j] > dist:
                P[j] = dist
            
            for i in range(1, len(T) - l + 1 - k):
                j = i+k
                dist = self.incrementalCompare(dist, T[i-1], T[j-1], T[i+l-1], T[j+l-1])
                if P[i] > dist:
                    P[i] = dist
                if P[j] > dist:
                    P[j] = dist
        
        scores = np.zeros(len(time_series))

        for i,p in enumerate(P):
            scores[i:i+l] = np.maximum(scores[i:i+l], p)


        return pd.DataFrame(scores, index=time_series.index, columns=["anom_score"])


    def compare(self, x1, x2):
        squaredSum = 0
        for v1, v2 in zip(x1, x2):
            squaredSum += (v1-v2)*(v1-v2)

        return math.sqrt(squaredSum)

    def incrementalCompare(self, dif, x1o, x2o, x1n, x2n):
        return math.sqrt(max(0,dif*dif - (x1o-x2o)*(x1o-x2o) + (x1n-x2n)*(x1n-x2n)))

    def subsequence(self, T, s, l):
        return T[s: s+l]
