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


class StatThresholdConfig(DetectorConfig):
 
    _default_transform = TemporalResample(granularity=None)

    _default_threshold = Threshold(alm_threshold=3.0)

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

class StatThreshold(DetectorBase):

    config_class = StatThresholdConfig

    # By default, we would like to train the model's post-rule (i.e. the threshold
    # at which we fire an alert) to maximize F1 score
    _default_post_rule_train_config = dict(metric=TSADMetric.F1)

    @property
    def require_even_sampling(self) -> bool:
        return True

    @property
    def require_univariate(self) -> bool:
        return True

    def __init__(self, config: StatThresholdConfig):

        super().__init__(config)



    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:
        pass

        

    def _get_anomaly_score(self, time_series: pd.DataFrame, time_series_prev: pd.DataFrame = None) -> pd.DataFrame:
        pass


