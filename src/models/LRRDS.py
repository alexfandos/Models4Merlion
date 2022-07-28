from merlion.models.anomaly.base import DetectorConfig
from merlion.post_process.threshold import Threshold
from merlion.evaluate.anomaly import TSADMetric
from merlion.models.anomaly.base import DetectorBase
from merlion.transform.resample import TemporalResample
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans


class LRRDSConfig(DetectorConfig):
 
    _default_transform = TemporalResample(granularity=None)

    _default_threshold = Threshold(alm_threshold=3.0)

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

class LRRDS(DetectorBase):

    config_class = LRRDSConfig

    # By default, we would like to train the model's post-rule (i.e. the threshold
    # at which we fire an alert) to maximize F1 score
    _default_post_rule_train_config = dict(metric=TSADMetric.F1)

    @property
    def require_even_sampling(self) -> bool:
        return True

    @property
    def require_univariate(self) -> bool:
        return False

    def __init__(self, config: LRRDSConfig):

        super().__init__(config)



    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:
        lrrds = LRRDSModel()
        return lrrds.process(train_data, 10)

        

    def _get_anomaly_score(self, time_series: pd.DataFrame, time_series_prev: pd.DataFrame = None) -> pd.DataFrame:
        lrrds = LRRDSModel()
        return lrrds.process(time_series, 10)

class LRRDSModel():

    def process(self, dataset, compresion_factor = 10):
        normalizedData = self.normalize(dataset)
        compressedData = self.compress(normalizedData, compresion_factor)
        ps = self.phaseSpaceReconstruction(compressedData, 10)
        score = 100000
        w = 2
        _, _, rm = self.localRM(ps, 10)
        for i in range(2, 10):
            
            lrec = self.LREC(rm, i)
            blrec = self.binarizeLREC(lrec)
            ss = self.getSegmentStarts(blrec)
            q = self.constructLengthMeanMatrix(lrec, ss)

            sc = sum(np.power(q[:,0]-np.mean(q[:,0]),2))/len(q)
            if sc < score:
                score = sc
                w = i
        lrec = self.LREC(rm, w)
        blrec = self.binarizeLREC(lrec)
        ss = self.getSegmentStarts(blrec)
        q = self.constructLengthMeanMatrix(lrec, ss)
        _, _, OD = self.RM(q)
        gdd = self.globalDiscordDegree(OD)
        ds = self.detectDiscordSegments(gdd)
        return self.constructDiscordDataframe(dataset.index, ss, ds, compresion_factor)

    def normalize(self, dataset):
        return dataset.apply(lambda x: 1 + (x - min(x)) / ( max(x) - min(x)), axis = 0, result_type = 'expand')

    def compress(self, dataset, compress_factor = 10):
        np_dataset = dataset.to_numpy()
        newSize = (math.ceil(np_dataset.shape[0]/float(compress_factor)), np_dataset.shape[1])
        compressed_dataset = np.zeros((newSize))
        for i in range(newSize[0]):
            s = i*compress_factor
            e = (i+1)*compress_factor
            if e > len(np_dataset):
                e = len(np_dataset)
            compressed_dataset[i] = np.mean(np_dataset[s:e], axis = 0)
        return compressed_dataset

    def phaseSpaceReconstruction(self, data,  time_delay = 1):
        M = len(data)
        res = np.zeros((M, data.shape[1]))
        for i in range(M):
            res[i,:] = data[i,:]
        return res



    def euclideanDistance(self, x1, x2):
        return math.sqrt(sum(np.square(x1 - x2)))

    def bhattacharyyaDistance(self, x1, x2):
        return math.sqrt(1-sum(x1*x2)/(sum(x1)*sum(x2)))


    def localRM(self, data, window_size):
        maxEuclidean = 0
        maxBhattacharyya = 0
        DE = np.zeros((data.shape[0], data.shape[0]))
        DB = np.zeros((data.shape[0], data.shape[0]))


        for x in range(window_size):
            for y in range(window_size):
                x1 = data[x,:]
                x2 = data[y,:]
                DE[x,y] = self.euclideanDistance(x1,x2)
                if DE[x,y] > maxEuclidean:
                    maxEuclidean = DE[x,y]
                DB[x,y] = self.bhattacharyyaDistance(x1,x2)
                if DB[x,y] > maxBhattacharyya:
                    maxBhattacharyya = DB[x,y]


        for i in range(1, len(data)+1-window_size):
            for j in range(i, i + window_size):
                x = j
                y = i + window_size - 1

                x1 = data[x,:]
                x2 = data[y,:]

                de = self.euclideanDistance(x1,x2)
                db = self.bhattacharyyaDistance(x1,x2)
                if de > maxEuclidean:
                    maxEuclidean = de
                if db > maxBhattacharyya:
                    maxBhattacharyya = db

                DE[x,y] = DE[y,x] = de
                DB[x,y] = DB[y,x] = db

        f = lambda i, th: i >= 0.25 * th
        DE = f(DE, maxEuclidean)
        DB = f(DB, maxBhattacharyya)

        return DE, DB, np.logical_and(DE, DB)

    def RM(self, data):
        maxEuclidean = 0
        maxBhattacharyya = 0
        DE = np.zeros((data.shape[0], data.shape[0]))
        DB = np.zeros((data.shape[0], data.shape[0]))
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                x1 = data[i,:]
                x2 = data[j,:]
                DE[i,j] = self.euclideanDistance(x1,x2)
                if DE[i,j] > maxEuclidean:
                    maxEuclidean = DE[i,j]
                DB[i,j] = self.bhattacharyyaDistance(x1,x2)
                if DB[i,j] > maxBhattacharyya:
                    maxBhattacharyya = DB[i,j]
        f = lambda x, y: x >= 0.25 * y
        DE = f(DE, maxEuclidean)
        DB = f(DB, maxBhattacharyya)

        return DE, DB, np.logical_and(DE, DB)


    def LREC(self, rm, w=4):
        LREC = np.zeros(rm.shape[0]-w)
        for i in range(rm.shape[0]-w):
            sum = 0
            for j in range(w):
                for k in range(w):
                    sum += rm[i+j, i+k]
            LREC[i] = 1/(w*w)*sum 
        return LREC

    def binarizeLREC(self, LREC):
        return LREC > 0


    def getSegmentStarts(self, blrec):
        segmentStarts = [0]
        for i in range(len(blrec)-1):
            if blrec[i] != blrec[i+1]:
                segmentStarts.append(i+1)
        return segmentStarts

    def constructLengthMeanMatrix(self, lrec, ss):
        Q = np.zeros((len(ss), 2))
        for i in range(len(ss)):
            start = ss[i]
            end = ss[i+1]  if i != len(ss)-1 else len(lrec) + 1
            Q[i,0] = end - start
            Q[i,1] = np.mean(lrec[start:end])
        return Q


    def obtainDistanceMatrixes(self, q):
        ED = BD = np.zeros((len(q), len(q)))
        maxEuclidean = 0
        maxBhattacharyya = 0
        for i in range(len(q)):
            for j in range(len(q)):
                ED[i,j] = self.euclideanDistance(q[i,:], q[j,:])
                BD[i,j] = self.bhattacharyyaDistance(q[i,:], q[j,:])


    def globalDiscordDegree(self, OD):
        gdd = np.zeros(len(OD))
        for i in range(len(OD)):
            gdd[i] = np.sum(OD[i,:]) / (len(OD) - 1)
        return gdd

    def detectDiscordSegments(self, gdd):
        kmeans = KMeans(n_clusters=2, random_state=0).fit(gdd.reshape(-1, 1))
        if np.mean(gdd[kmeans.labels_ == 1]) > np.mean(gdd[kmeans.labels_ == 0]):
            return kmeans.labels_
        else:
            return np.abs(kmeans.labels_ - 1)

    def constructDiscordDataframe(self, timestamps, ss, ds, compression_rate):
        df = pd.DataFrame(index = timestamps)
        df["anomalyScore"] = 0
        for i in range(len(ss)):
            if ds[i] == 1:
                start = ss[i]*compression_rate
                end = ss[i+1]*compression_rate  if i != len(ss)-1 else len(timestamps) + 1
                df.iloc[start:end] = 1
        return df
