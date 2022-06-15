import matplotlib.pyplot as plt
import numpy as np

from merlion.plot import plot_anoms
from merlion.utils import TimeSeries
from ts_datasets.anomaly import NAB
from ts_datasets.anomaly import MSL
from merlion.transform.moving_average import MovingAverage


from models.SlidingWindowARIMA import StatThresholdConfig, StatThreshold

# This is a time series with anomalies in both the train and test split.
# time_series and metadata are both time-indexed pandas DataFrames.
time_series, metadata = NAB(subset="realKnownCause")[3]


#time_series, metadata = MSL()[0]

# Get training split
train = time_series[metadata.trainval]
train_data = TimeSeries.from_pd(train)
train_labels = TimeSeries.from_pd(metadata[metadata.trainval].anomaly)





# Get testing split
test = time_series[~metadata.trainval]
test_data = TimeSeries.from_pd(test)
test_labels = TimeSeries.from_pd(metadata[~metadata.trainval].anomaly)





# Initialize a model & train it. The dataframe returned & printed
# below is the model's anomaly scores on the training data.
model = StatThreshold(StatThresholdConfig())
model.train(train_data=train_data, anomaly_labels=test_labels)


# Let's run the our model on the test data, both with and without
# applying the post-rule. Notice that applying the post-rule filters out
# a lot of entries!
import pandas as pd
anom_scores = model.get_anomaly_score(test_data).to_pd()
anom_labels = model.get_anomaly_label(test_data).to_pd()
print(pd.DataFrame({"no post rule": anom_scores.iloc[:, 0],
                    "with post rule": anom_labels.iloc[:, 0]}))


                    # Additionally, notice that the nonzero post-processed anomaly scores,
# are interpretable as z-scores. This is due to the automatic calibration.
print(anom_labels[anom_labels.iloc[:, 0] != 0])


print("no post rule")
fig, ax = model.plot_anomaly(test_data, filter_scores=False)
plot_anoms(ax, test_labels)
plt.show()


print("with post rule")
fig, ax = model.plot_anomaly(test_data, filter_scores=True)
plot_anoms(ax, test_labels)
plt.show()