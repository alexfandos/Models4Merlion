from merlion.models.anomaly.base import DetectorConfig
from merlion.post_process.threshold import Threshold
from merlion.evaluate.anomaly import TSADMetric
from merlion.models.anomaly.base import DetectorBase
from merlion.transform.resample import TemporalResample

import pandas as pd
import numpy as np
import torch
import math


class DeepAnTConfig(DetectorConfig):
 
    _default_transform = TemporalResample(granularity=None)

    # When you call model.get_anomaly_label(), you will transform the model's
    # raw anomaly scores (returned by model.get_anomaly_score() into z-scores,
    # and you will apply a thresholding rule to suppress all anomaly scores
    # with magnitude smaller than the threshold. Here, we only wish to report
    # 3-sigma events.
    _default_threshold = Threshold(alm_threshold=3.0)

    def __init__(self, windows_size = 20, **kwargs):
        """
        Provide model-specific config parameters here, with kwargs to capture any
        general-purpose arguments used by the base class. For DetectorConfig,
        these are transform and post_rule.
        
        We include the initializer here for clarity. In this case, it may be
        excluded, as it only calls the superclass initializer.
        """
        super().__init__(**kwargs)
        self.windows_size = windows_size


class DeepAnT(DetectorBase):
    # The config class for StatThreshold is StatThresholdConfig, defined above
    config_class = DeepAnTConfig

    # By default, we would like to train the model's post-rule (i.e. the threshold
    # at which we fire an alert) to maximize F1 score
    _default_post_rule_train_config = dict(metric=TSADMetric.F1)

    @property
    def require_even_sampling(self) -> bool:
        return True

    @property
    def require_univariate(self) -> bool:
        return False

    def __init__(self, config: DeepAnTConfig):

        super().__init__(config)
        self.model = None

    def __make_train_step(self, model, loss_fn, optimizer):
        """
            Computation : Function to make batch size data iterator
        """
        def train_step(x, y):
            model.train()
            yhat = model(x)
            loss = loss_fn(y, yhat)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            return loss.item()
        return train_step

    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:

        X, Y = self.data_pre_processing(train_data)

        # create model
        self.model = DeepAnTModel(self.config.windows_size, self.dim)
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(list(self.model.parameters()), lr=1e-5)

        train_data_tensor = torch.utils.data.TensorDataset(torch.tensor(X.astype(np.float32)), torch.tensor(Y.astype(np.float32)))

        train_loader = torch.utils.data.DataLoader(dataset=train_data_tensor, batch_size=32, shuffle=False)
        train_step = self.__make_train_step(self.model, criterion, optimizer)

        for epoch in range(20):
            loss_sum = 0.0
            ctr = 0
            for x_batch, y_batch in train_loader:
                loss_train = train_step(x_batch, y_batch)
                loss_sum += loss_train
                ctr += 1
            print("Training Loss: {0} - Epoch: {1}".format(float(loss_sum/ctr), epoch+1))
        
        processed_anomalies = (self.model(torch.tensor(X.astype(np.float32))).detach().numpy()-Y)

        

        anomalies = np.concatenate((np.zeros((self.config.windows_size, self.dim)), processed_anomalies))

        loss = np.linalg.norm(anomalies, axis=1)

        return pd.DataFrame(loss, index=train_data.index, columns=["anom_score"])

    def _get_anomaly_score(self, time_series: pd.DataFrame, time_series_prev: pd.DataFrame = None) -> pd.DataFrame:
        
        X, Y = self.data_pre_processing(time_series)

        processed_anomalies = (self.model(torch.tensor(X.astype(np.float32))).detach().numpy()-Y)

        anomalies = np.concatenate((np.zeros((self.config.windows_size, self.dim)), processed_anomalies))

        loss = np.linalg.norm(anomalies, axis=1)

        return pd.DataFrame(loss, index=time_series.index, columns=["anom_score"])


    def data_pre_processing(self, df):
        """
            Data pre-processing : Function to create data for Model
        """
        try:
            #df = df.to_frame()
            _data_ = df.to_numpy(copy=True)  #create a numpy copy
            X = np.zeros(shape=(df.shape[0]-self.config.windows_size,self.config.windows_size,df.shape[1]))
            Y = np.zeros(shape=(df.shape[0]-self.config.windows_size,self.dim))
            for i in range(self.config.windows_size-1, df.shape[0]-1):
                Y[i-self.config.windows_size+1] = _data_[i+1]
                for j in range(i-self.config.windows_size+1, i+1):
                    X[i-self.config.windows_size+1][self.config.windows_size-1-i+j] = _data_[j]
            X = X.transpose((0, 2, 1))
            return X,Y
        except Exception as e:
            print("Error while performing data pre-processing : {0}".format(e))
            return None, None, None


class DeepAnTModel(torch.nn.Module):
    """
        Model : Class for DeepAnT model
    """
    def __init__(self, LOOKBACK_SIZE, CHANNELS):
        super(DeepAnTModel, self).__init__()
        self.conv1d_1_layer = torch.nn.Conv1d(in_channels=CHANNELS, out_channels=32, kernel_size=3)
        self.relu_1_layer = torch.nn.ReLU()
        self.maxpooling_1_layer = torch.nn.MaxPool1d(kernel_size=2)
        self.conv1d_2_layer = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)
        self.relu_2_layer = torch.nn.ReLU()
        self.maxpooling_2_layer = torch.nn.MaxPool1d(kernel_size=2)
        self.flatten_layer = torch.nn.Flatten()
        self.dense_1_layer = torch.nn.Linear(32*math.floor((math.floor((LOOKBACK_SIZE-2)/2)-2)/2), 40)
        self.relu_3_layer = torch.nn.ReLU()
        self.dropout_layer = torch.nn.Dropout(p=0.25)
        self.output = torch.nn.Linear(40, CHANNELS)
        
    def forward(self, x):
        x = self.conv1d_1_layer(x)   #[batch_size, 32,  ws-2]
        x = self.relu_1_layer(x)    
        x = self.maxpooling_1_layer(x)   #[batch_size, 32,  (ws-2)/2]
        x = self.conv1d_2_layer(x)   #[batch_size, 32,  ((ws-2)/2)-2]
        x = self.relu_2_layer(x)
        x = self.maxpooling_2_layer(x)  #[batch_size, 32,  (((ws-2)/2)-2)/2]
        x = self.flatten_layer(x)     #32*(((ws-2)/2)-2)/2
        x = self.dense_1_layer(x)
        x = self.relu_3_layer(x)
        x = self.dropout_layer(x)
        return self.output(x)