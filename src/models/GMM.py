from merlion.models.anomaly.base import DetectorConfig
from merlion.post_process.threshold import Threshold
from merlion.evaluate.anomaly import TSADMetric
from merlion.models.anomaly.base import DetectorBase
from merlion.transform.resample import TemporalResample
from merlion.utils import TimeSeries
from datetime import timedelta
from sklearn.mixture import GaussianMixture
from sklearn import metrics

import pandas as pd
import numpy as np
import math


class StatThresholdConfig(DetectorConfig):
 
    _default_transform = TemporalResample(granularity=None)

    _default_threshold = Threshold(alm_threshold=3.0)

    def __init__(self, seasonality = timedelta(days = 1), bins = 12, f = 2, **kwargs):

        self.f = f
        self.seasonality = seasonality
        self.bins = bins

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
        self.firstTimeStamp = train_data.index[0]
        ScOS = self.trainGMM(train_data)
        valuesWithoutOutliers = np.zeros(len(train_data))
        for i in range(len(valuesWithoutOutliers)):
            valuesWithoutOutliers[i] = self.removeOutlier(train_data.values[i], train_data.index[i], ScOS[i])

        train_data_without_anomalies = pd.DataFrame(index = train_data.index)
        train_data_without_anomalies["value"] = valuesWithoutOutliers

        ScOS = self.trainGMM(train_data_without_anomalies)

        df = pd.DataFrame(index = train_data.index)
        df["anomalyScore"] = ScOS
        return df


    def trainGMM(self, train_data: pd.DataFrame):
        self.bins = self.createBins(train_data)
        self.gmm = self.generateGMM(self.bins)
        OSc = np.zeros(len(train_data))
        for i in range(len(OSc)):
            OSc[i] = self.obtainOutlierScore(train_data.values[i], train_data.index[i])
        # OS = np.vectorize(self.obtainOutlierScore)(train_data.values, train_data.index)
        self.maxOS = max(OSc)
        self.minOS = min(OSc)

        return np.vectorize(self.obtainScaledOutlierScore)(OSc)
            

    def _get_anomaly_score(self, time_series: pd.DataFrame, time_series_prev: pd.DataFrame = None) -> pd.DataFrame:
        OSc = np.zeros(len(time_series))
        for i in range(len(OSc)):
            OSc[i] = self.obtainOutlierScore(time_series.values[i], time_series.index[i])

        ScOS = np.vectorize(self.obtainScaledOutlierScore)(OSc)

        df = pd.DataFrame(index = time_series.index)
        df["anomalyScore"] = ScOS
        return df

    def createBins(self, train_Data):
        bins = [[] for x in range(self.config.bins)]

        for i in range(len(train_Data)):
            bins[self.getBinNumber(train_Data.index[i])].append(train_Data.value[i])
        return bins

    def getBinNumber(self, time_stamp):
        return math.floor((time_stamp - self.firstTimeStamp) /self.config.seasonality % 1 * self.config.bins)

    def generateGMM(self, bins):
        gmm = [None for x in range(self.config.bins)]
        for i in range(self.config.bins):

                data = np.array(bins[i]).reshape(-1,1)
                gmm[i] = GaussianMixture(n_components=2).fit(data)
                # labels = gmm_n.predict(data)
                # ss = metrics.silhouette_score(data, labels, metric='euclidean')
                # if ss > best:
                #     best = ss
                #     gmm[i] = gmm_n
        return gmm

    def obtainOutlierScore(self, value, time_stamp):
        w = self.gmm[self.getBinNumber(time_stamp)].weights_
        m = self.gmm[self.getBinNumber(time_stamp)].means_
        c = self.gmm[self.getBinNumber(time_stamp)].covariances_
        w = w.reshape(len(w))
        m = m.reshape(len(m))
        c = c.reshape(len(c))
        p = self.p(value, w, m, c)
        return math.pow(math.log(p+0.00001), 2*self.config.f)


    def obtainScaledOutlierScore(self, OSc):
        return (OSc - self.minOS)/(self.maxOS-self.minOS)*10
        

    def p(self, x, w, m, c):
        sum = 0
        for i in range(len(w)):
            g = 1/math.sqrt(2*math.pi*c[i]*c[i])*math.pow(math.e, - math.pow(x-m[i],2)/(2*c[i]*c[i]))
            sum += w[i]*g
        return sum

    def removeOutlier(self, train_data, time_stamp, scOS):
        if scOS >= 8:
            w = self.gmm[self.getBinNumber(time_stamp)].weights_
            m = self.gmm[self.getBinNumber(time_stamp)].means_
            return m[np.argmax(w)]
        else:
            return train_data
        


class SlidingWindowARIMA():

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


