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


class PAPRRWConfig(DetectorConfig):
 
    _default_transform = TemporalResample(granularity=None)

    _default_threshold = Threshold(alm_threshold=1.2)

    def __init__(self, segment_length = 430, sub_value_space_number = 6, wd=0.3, wc=0.4, wr = 0.3, damping_factor = 0.1, steps = 200, **kwargs):
        
        self.damping_factor = damping_factor
        self.steps = steps

        self.wd = wd / (wd + wc + wr)
        self.wc = wc / (wd + wc + wr)
        self.wr = wr / (wd + wc + wr)
        
        self.segment_length = segment_length
        self.sub_value_space_number = sub_value_space_number
        super().__init__(**kwargs)

class PAPRRW(DetectorBase):

    config_class = PAPRRWConfig

    # By default, we would like to train the model's post-rule (i.e. the threshold
    # at which we fire an alert) to maximize F1 score
    _default_post_rule_train_config = dict(metric=TSADMetric.F1)

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return True

    def __init__(self,  config: PAPRRWConfig):

        super().__init__(config)



    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:
        return self.process(train_data)

        

    def _get_anomaly_score(self, time_series: pd.DataFrame, time_series_prev: pd.DataFrame = None) -> pd.DataFrame:
        return self.process(time_series)

    
    def process(self, time_series):
        self.entropy = self.estimateEntropy(time_series.to_numpy())
        segments = self.segmentate(time_series)
        M = []
        for segment in segments:
            epsvs = self.estimateEqualProbabilitySubValueSpaces(segment)
            m =  self.generateM(segment, epsvs)
            M.append(m)

        M = np.asarray(M)
        S = self.constructSimilarityMatrix(M)
        c = np.ones(len(segments))/len(segments)
        c = self.randomWalk(c, S, self.config.damping_factor, self.config.steps)
        segmentScores = 1-c
        maxSS = np.max(segmentScores)
        minSS = np.min(segmentScores)

        segmentScores = (segmentScores - minSS) / (maxSS-minSS)


        scores = []
        for i, segment in enumerate(segments):
            scores += len(segment) * [segmentScores[i]]
        scores = np.asarray(scores)


        return pd.DataFrame(scores, index=time_series.index, columns=["anom_score"])


    def segmentate(self, time_series):
        splits = math.ceil(len(time_series)/self.config.segment_length)
        return np.array_split(time_series.to_numpy(), splits)
        
    def estimateEqualProbabilitySubValueSpaces(self, segment):
        m = np.mean(segment)
        std = np.std(segment)
        min = np.min(segment)
        max = np.max(segment)

        

        minCP = norm(loc=m, scale=std).cdf(min)
        maxCP = norm(loc=m, scale=std).cdf(max)

        intervalCP = maxCP - minCP

        subValueProbability = intervalCP / self.config.sub_value_space_number

        values = np.linspace(min, max, len(segment)*self.config.sub_value_space_number)

        lastCPChangeValue = minCP

        changeValues = np.asarray([min])

        for value in values:
            if (norm(loc=m, scale=std).cdf(value) - lastCPChangeValue >= subValueProbability):
                changeValues = np.append(changeValues, value)
                lastCPChangeValue = norm(loc=m, scale=std).cdf(value)

        return changeValues


    def getPointFeatureVector(self, segment, subValueSpace):
        d = np.zeros(len(subValueSpace))
        for i in range(len(d)-1):
            d[i]  = sum(((segment >= subValueSpace[i]) & (segment < subValueSpace[i+1])))
        d[len(d)-1] =  sum(segment >= subValueSpace[len(d)-1])
        return d

    def getCenterFeatureVector(self, segment, subValueSpace):
        c = np.zeros(len(subValueSpace))
        for i in range(len(c)-1):
            c[i]  = np.mean(segment[(segment >= subValueSpace[i]) & (segment < subValueSpace[i+1])])
            if math.isnan(c[i]):
                c[i] = (subValueSpace[i] + subValueSpace[i+1])/2

        c[len(c)-1] =  np.mean(segment[(segment >= subValueSpace[len(c)-1])])
        if math.isnan(c[len(c)-1]):
                c[len(c)-1] = subValueSpace[c[len(c)-1]] 
        return c

    def getDispersionFeatureVector(self, segment, subValueSpace):
        r = np.zeros(len(subValueSpace))
        for i in range(len(r)-1):
            r[i]  = np.var(segment[(segment >= subValueSpace[i]) & (segment < subValueSpace[i+1])])
            if math.isnan(r[i]):
                r[i] = 1
        r[len(r)-1] =  np.var(segment[(segment >= subValueSpace[len(r)-1])])
        if math.isnan(r[len(r)-1]):
                r[len(r)-1] = 1
        return r

    def generateM(self, segment, subValueSpace):
        return {"d" : self.getPointFeatureVector(segment, subValueSpace), "c" : self.getCenterFeatureVector(segment, subValueSpace),"r" : self.getDispersionFeatureVector(segment, subValueSpace)}


    def estimateEntropy(self, time_series):
        divisions = math.ceil(len(time_series)/10)
        minV = min(time_series)
        maxV = max(time_series)
        if minV == maxV:
            return 0
        p = np.zeros(divisions)
        steps = np.linspace(minV, maxV, divisions+1)
        for i in range(divisions-1):
            p[i] = sum(((time_series >= steps[i]) & (time_series < steps[i+1]))) / len(time_series)
        p[len(p)-1] = sum((time_series >= steps[len(p)-1])) / len(time_series)
        return (entropy(p))

    def dSimilarity(self, d1, d2):
        return np.sum(np.minimum(d1, d2)) / np.sum(d1)

    def grbsSimilarity(self, x1, x2):
        sigma = self.entropy
        return math.exp( - np.sum ( np.square ( x1 - x2 ) ) / (sigma*sigma))

    def constructSimilarityMatrix(self, M):
        S = np.zeros((len(M), len(M)))
        for i in range(len(M)-1):
            for j in range(i+1, len(M)):
                ds = self.dSimilarity(M[i]["d"], M[j]["d"])
                cs = self.dSimilarity(M[i]["c"], M[j]["c"])
                rs = self.dSimilarity(M[i]["r"], M[j]["r"])
                s = self.config.wd * ds + self.config.wc * cs + self.config.wr * rs
                S[i,j] = S[j,i] = s

        return S
                
    def randomWalk(self, c, S, d, steps):

        S_ = S / np.sum(S, axis = 1).reshape(-1,1)

        for i in range(steps):
            c = d*c + np.matmul(c,(1 - d)*S_)
        return c