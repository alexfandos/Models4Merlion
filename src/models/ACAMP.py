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


class ACAMPConfig(DetectorConfig):
 
    _default_transform = TemporalResample(granularity=None)

    _default_threshold = Threshold(alm_threshold=4)

    def __init__(self, segment_length = 290, minimal_compare_distance = 290, **kwargs):
        
        self.segment_length = segment_length
        self.minimal_compare_distance = minimal_compare_distance
        super().__init__(**kwargs)

class ACAMP(DetectorBase):

    config_class = ACAMPConfig

    # By default, we would like to train the model's post-rule (i.e. the threshold
    # at which we fire an alert) to maximize F1 score
    _default_post_rule_train_config = dict(metric=TSADMetric.F1)

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return True

    def __init__(self,  config: ACAMPConfig):

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

            ti = self.subsequence(T, i, l)
            tj = self.subsequence(T, j, l)

            A = self.sum(ti)
            B = self.sum(tj)
            sA = self.squaredSum(ti)
            sB = self.squaredSum(tj)
            C = self.productSum(ti, tj)

            dist = self.dist(l, A, B, sA, sB, C)


            if P[i] > dist:
                P[i] = dist
            if P[j] > dist:
                P[j] = dist
            
            for i in range(1, len(T) - l + 1 - k):
                j = i+k

                tio = T[i-1]
                tjo = T[j-1]
                tin = T[i+l-1]
                tjn = T[j+l-1]

                A = self.incrementalSum(A, tio, tin)
                B = self.incrementalSum(A, tjo, tjn)
                sA = self.incrementalSquaredSum(sA, tio, tin)
                sB  = self.incrementalSquaredSum(sB, tjo, tjn)
                C = self.incrementalProductSum(C, tio, tjo, tin, tjn)

                if sA-(1/l)*A*A < 0:
                    sA = (1/l)*A*A + 0.001


                if sB-(1/l)*B*B < 0:
                    sB = (1/l)*B*B + 0.001


                dist = self.dist(l, A, B, sA, sB, C)


                if P[i] > dist:
                    P[i] = dist
                if P[j] > dist:
                    P[j] = dist
        
        scores = np.zeros(len(time_series))

        for i,p in enumerate(P):
            scores[i:i+l] = np.maximum(scores[i:i+l], p)


        return pd.DataFrame(scores, index=time_series.index, columns=["anom_score"])


    def sum(self, x):
        return np.sum(x)

    def squaredSum(self, x):
        return np.sum(x**2)
    
    def productSum(self, x1, x2):
        return np.sum(x1*x2)

    def incrementalSum(self, sum, xo, xn):
        return sum - xo + xn

    def incrementalSquaredSum(self, squaredSum, xo, xn):
        return squaredSum - xo*xo + xn*xn

    def incrementalProductSum(self, productSum, x1o, x2o, x1n, x2n):
        return productSum - x1o*x2o + x1n*x2n

    def dist(self, m, A,B, sA, sB, C):
        return 2*m*(1-(C-A*B/m) / math.sqrt((sA-1/m*A*A)*(sB-1/m*B*B)))

    def subsequence(self, T, s, l):
        return T[s: s+l]
