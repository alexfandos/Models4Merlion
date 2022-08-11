from doctest import OutputChecker
from logging import root
from time import time
from merlion.models.anomaly.base import DetectorConfig
from merlion.post_process.threshold import Threshold
from merlion.evaluate.anomaly import TSADMetric
from merlion.models.anomaly.base import DetectorBase
from merlion.transform.resample import TemporalResample
from statsmodels.tsa.stattools import adfuller
from merlion.utils import TimeSeries


import pandas as pd
import numpy as np
import math
import torch



class EnrichedCNNAEConfig(DetectorConfig):
 
    _default_transform = TemporalResample(granularity=None)

    _default_threshold = Threshold(alm_threshold=4.5)

    def __init__(self, window_size_b = 290, window_size_f = 2, n_epochs = 1000, learning_rate = 0.001,  **kwargs):
        self.b = window_size_b
        self.f = window_size_f
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        super().__init__(**kwargs)

class EnrichedCNNAE(DetectorBase):

    config_class = EnrichedCNNAEConfig

    # By default, we would like to train the model's post-rule (i.e. the threshold
    # at which we fire an alert) to maximize F1 score
    _default_post_rule_train_config = dict(metric=TSADMetric.F1)

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return True

    def __init__(self, config: EnrichedCNNAEConfig):
        self.model = EnrichedCNNAEModel(b = config.b, f = config.f, n_epochs=config.n_epochs)
        super().__init__(config)



    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:
        self.model.setK(len(train_data.columns))
        return self.model.train(train_data, self.config.learning_rate)

        

    def _get_anomaly_score(self, time_series: pd.DataFrame, time_series_prev: pd.DataFrame = None) -> pd.DataFrame:
        return self.model.test(time_series)


class EnrichedCNNAEModel():
    def __init__(self, b, f, n_epochs):
        if b % 2 != 0:
            b = b + 1
        self.b = b
        if f % 2 != 0:
            f = f + 1
        self.f = f
        self.model = AutoEncoder()
        self.n_epochs = n_epochs

    def train(self, time_series: pd.DataFrame, learning_rate = 0.001):
        enrichedData = self.enrichData(time_series)

        enrichedData = self.expandDimensionIfToSmall(enrichedData)

        train_data_tensor = torch.utils.data.TensorDataset(torch.tensor(enrichedData.astype(np.float32)), torch.tensor(enrichedData.astype(np.float32)))

        train_loader = torch.utils.data.DataLoader(dataset=train_data_tensor, batch_size=32, shuffle=False)

        self.trainModel(train_loader, learning_rate)

        error = self.predict(enrichedData)

        return self.buildAnomalyScores(time_series.index, error)

    def test(self, time_series: pd.DataFrame):
        enrichedData = self.enrichData(time_series)

        enrichedData = self.expandDimensionIfToSmall(enrichedData)

        error = self.predict(enrichedData)

        return self.buildAnomalyScores(time_series.index, error)

    
    def predict(self, data):
        raw_predictions = self.model(torch.tensor(data.astype(np.float32))).detach().numpy()
        error = raw_predictions - data
        square_error = np.square(error)
        mean_square_error = np.mean(square_error, axis = (1,2,3))
        root_mean_square_error = np.sqrt(mean_square_error)
        return root_mean_square_error
        
    
    def setK(self, k):
        self.k = k
    
    def enrichData(self, time_series: pd.DataFrame):
        return self.buildHVector(self.buildGVector(time_series))

    def expandDimensionIfToSmall(self, dataset):
        if self.k >= 16:
            return dataset
        
        while(dataset.shape[3] < 16):
            dataset = np.concatenate((dataset, dataset), axis=3)

        return dataset


    def buildAnomalyScores(self, indexes, errors):
        points = len(indexes)
        xp = np.linspace(0, points-1, len(errors))
        x = np.linspace(0, points-1, points)
        anomalies = np.interp(x, xp, errors)

        return pd.DataFrame(anomalies, index=indexes, columns=["anom_score"])

    def buildGVector(self, time_series: pd.DataFrame):
        g_length = math.ceil(len(time_series) / (self.b / 2))
        G = np.zeros((g_length, 2, self.k))

        for i in range(g_length):
            s = int(i * self.b / 2)
            e = int(i * self.b / 2 + self.b)
            if e > len(time_series):
                e = len(time_series)
            
            windows = time_series.iloc[s:e].to_numpy()

            G[i,0,:] = np.sqrt(np.sum(np.square(windows), axis=0))

            if i != 0:
                G[i,1,:] = G[i,0,:] - G[i-1,0,:]
        return G
    
    def buildHVector(self, G):
        h_length = math.ceil(len(G) / (self.f / 2))
        H = np.zeros((h_length, 1, 16, self.k))
        for i in range(h_length):
            s = int(i * self.f / 2)
            e = int(i * self.f / 2 + self.f)
            windowsNOR = G[s:e, 0, :]
            windowsDON = G[s:e, 1, :]
            H[i,0,0,:] = np.mean(windowsNOR, axis=0)
            H[i,0,1,:] = np.min(windowsNOR, axis = 0)
            H[i,0,2,:] = np.max(windowsNOR, axis = 0)
            H[i,0,3,:] = np.quantile(windowsNOR, 0.25, axis=0)
            H[i,0,4,:] = np.quantile(windowsNOR, 0.5, axis=0)
            H[i,0,5,:] = np.quantile(windowsNOR, 0.75, axis=0)
            H[i,0,6,:] = np.std(windowsNOR, axis = 0)
            H[i,0,7,:] = np.ptp(windowsNOR, axis = 0)

            H[i,0,8,:] = np.mean(windowsDON, axis=0)
            H[i,0,9,:] = np.min(windowsDON, axis = 0)
            H[i,0,10,:] = np.max(windowsDON, axis = 0)
            H[i,0,11,:] = np.quantile(windowsDON, 0.25, axis=0)
            H[i,0,12,:] = np.quantile(windowsDON, 0.5, axis=0)
            H[i,0,13,:] = np.quantile(windowsDON, 0.75, axis=0)
            H[i,0,14,:] = np.std(windowsDON, axis = 0)
            H[i,0,15,:] = np.ptp(windowsDON, axis = 0)

        return H

    def trainModel(self, train_loader, learning_rate = 0.001):
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        self.model.to(device)

        n_epochs = self.n_epochs
        for epoch in range(1, n_epochs+1):
            train_loss = 0.0
            ctr = 0

            for data in train_loader:
                images, _ = data
                images = images.to(device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, images)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                ctr += 1
            train_loss = train_loss/ctr
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 2, padding='same')  

        self.conv2 = torch.nn.Conv2d(32, 16, 2, padding='same')  

        self.conv3 = torch.nn.Conv2d(16, 4, 2, padding='same')  

        self.pool = torch.nn.MaxPool2d(2, 2)

        self.t_conv1 = torch.nn.ConvTranspose2d(4, 16, 2, stride=2)

        self.t_conv2 = torch.nn.ConvTranspose2d(16, 32, 2, stride=2)

        self.t_conv3 = torch.nn.ConvTranspose2d(32, 1, 2, stride=2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.sigmoid(self.conv3(x))
        x = self.pool(x)

        x = torch.relu(self.t_conv1(x))
        x = torch.relu(self.t_conv2(x))
        x = self.t_conv3(x)
        
        return x