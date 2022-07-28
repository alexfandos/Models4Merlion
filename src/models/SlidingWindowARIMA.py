from merlion.models.anomaly.base import DetectorConfig
from merlion.post_process.threshold import Threshold
from merlion.evaluate.anomaly import TSADMetric
from merlion.models.anomaly.base import DetectorBase
from merlion.transform.resample import TemporalResample
from merlion.transform.moving_average import DifferenceTransform
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from merlion.models.forecast.arima import Arima, ArimaConfig
from merlion.utils import TimeSeries


import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans


class SlidingWindowARIMAConfig(DetectorConfig):
 
    _default_transform = TemporalResample(granularity=None)

    _default_threshold = Threshold(alm_threshold=3.0)

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

class SlidingWindowARIMA(DetectorBase):

    config_class = SlidingWindowARIMAConfig

    # By default, we would like to train the model's post-rule (i.e. the threshold
    # at which we fire an alert) to maximize F1 score
    _default_post_rule_train_config = dict(metric=TSADMetric.F1)

    @property
    def require_even_sampling(self) -> bool:
        return True

    @property
    def require_univariate(self) -> bool:
        return True

    def __init__(self, config: SlidingWindowARIMAConfig):

        super().__init__(config)



    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:
        lrrds = SlidingWindowARIMAModel()
        return lrrds.process(train_data)

        

    def _get_anomaly_score(self, time_series: pd.DataFrame, time_series_prev: pd.DataFrame = None) -> pd.DataFrame:
        lrrds = SlidingWindowARIMAModel()
        return lrrds.process(time_series)

class SlidingWindowARIMAModel():

    def process(self, dataset: pd.DataFrame, window_size = 500, step = 50):
        anomalyScores = np.zeros((len(dataset),1))

        for i in range(0, len(dataset) - window_size, step):
            print(i, "/", len(dataset) - window_size)
            sliding_window_data = dataset.iloc[i: i + window_size]
            D = self.obtainD(sliding_window_data)
            P, Q, trainedModel = self.obtainPandQ(sliding_window_data, D)
            timestamp = dataset.iloc[i+window_size+1:i+window_size+1+step].index

            pred, _ = trainedModel.forecast(time_stamps = timestamp)

            anomalyScores[i+window_size+1 : i+window_size+1+step] = pred.to_pd().to_numpy()

        df = pd.DataFrame(index = dataset.index)
        df["anomalyScore"] = anomalyScores
        return df



    def obtainD(self, dataset: pd.DataFrame):
        differentiated_sliding_window_data = dataset.copy(deep = True)
        D = 1
        differentiate = DifferenceTransform()

        dswd = TimeSeries.from_pd(differentiated_sliding_window_data)

        differentiate.train(dswd)
        if (len(dswd) < 100):
            a = 1
        dswd = differentiate(dswd)
        while adfuller(dswd.to_pd().to_numpy())[1] > 0.05:
            if (len(dswd) < 100):
                a = 1
            D = D + 1
            dswd = differentiate(dswd)
        return D

    def obtainPandQ(self, dataset: pd.DataFrame, D, Pmax = 2, Qmax = 2):
        dataset = TimeSeries.from_pd(dataset)
        minAICvalue = None
        minP = None
        minQ = None
        minModel = None
        for p in range(Pmax):
            for q in range(Qmax):
                model = Arima(ArimaConfig(order=(p,D,q)))
                _, train_err = model.train(train_data = dataset)
                if (len(train_err.to_pd().to_numpy()) == 0):
                    continue
                AICValue = self.AIC(train_err.to_pd().to_numpy(), p + q + 1)
                if minAICvalue == None or minAICvalue > AICValue:
                    minAICvalue = AICValue
                    minP = p
                    minQ = q
                    minModel = model
        return minP, minQ, model

    def AIC(self, error, K):
        return len(error) * math.log(sum(np.square(error))/len(error)) + K



