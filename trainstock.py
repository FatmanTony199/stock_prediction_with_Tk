import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import yfinance as yf
import pandas as pd
import numpy as np
import tkinter as tk
import time


from zipfile import ZipFile
import os

# uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
# zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
# zip_file = ZipFile(zip_path)
# zip_file.extractall()
# csv_path = "jena_climate_2009_2016.csv"

# df = pd.read_csv(csv_path)
company = '2330.TW'
msft = yf.Ticker(company)
hist = msft.history(period="max")
print(hist.iloc[-2:])

# 計算股票的布林通道
sma20 = hist["Close"].rolling(window=20).mean()
std20 = hist["Close"].rolling(window=20).std()
upper_band = sma20 + 2 * std20
lower_band = sma20 - 2 * std20
upper_band_s = sma20 + 1 * std20
lower_band_s = sma20 - 1 * std20

hist['sma20'] = sma20
hist['upper_band_s'] = upper_band_s
hist['upper_band'] = upper_band
hist['lower_band_s'] = lower_band_s
hist['lower_band'] = lower_band
#replace 'NaN' to 0
hist = hist.fillna(0)

print(hist.iloc[-20:])

"""
## Raw Data Visualization
To give us a sense of the data we are working with, each feature has been plotted below.
This shows the distinct pattern of each feature over the time period from 2009 to 2016.
It also shows where anomalies are present, which will be addressed during normalization.
"""

titles = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Dividends",
    "Stock Splits",
    "sma20",
    "upper_band_s",
    "upper_band",
    "lower_band",
    "lower_band_s",
]

feature_keys = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Dividends",
    "Stock Splits",
    "sma20",
    "upper_band_s",
    "upper_band",
    "lower_band",
    "lower_band_s",
]

colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]

date_time_key = "Date"


def show_raw_visualization(data):
    time_data = range(len(data))
    fig, axes = plt.subplots(
        nrows=6, ncols=2, figsize=(15, 20), dpi=80, facecolor="w", edgecolor="k"
    )
    for i in range(len(feature_keys)):
        key = feature_keys[i]
        c = colors[i % (len(colors))]
        t_data = data[key]
        t_data.index = time_data
        t_data.head()
        ax = t_data.plot(
            ax=axes[i // 2, i % 2],
            color=c,
            title="{} - {}".format(titles[i], key),
            rot=25,
        )
        ax.legend([titles[i]])
    plt.tight_layout()


show_raw_visualization(hist)

"""
This heat map shows the correlation between different features.
"""


def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()


show_heatmap(hist)


"""
## Data Preprocessing
Here we are picking ~300,000 data points for training. Observation is recorded every
10 mins, that means 6 times per hour. We will resample one point per hour since no
drastic change is expected within 60 minutes. We do this via the `sampling_rate`
argument in `timeseries_dataset_from_array` utility.
We are tracking data from past 720 timestamps (720/6=120 hours). This data will be
used to predict the temperature after 72 timestamps (72/6=12 hours).
Since every feature has values with
varying ranges, we do normalization to confine feature values to a range of `[0, 1]` before
training a neural network.
We do this by subtracting the mean and dividing by the standard deviation of each feature.
71.5 % of the data will be used to train the model, i.e. 300,693 rows. `split_fraction` can
be changed to alter this percentage.
The model is shown data for first 5 days i.e. 720 observations, that are sampled every
hour. The temperature after 72 (12 hours * 6 observation per hour) observation will be
used as a label.
"""

split_fraction = 1
train_split = int(split_fraction * int(hist.shape[0]))
step = 1
days = 5
past = 120
future = 0
learning_rate = 0.001
batch_size = 32
epochs = 20
data_mean =[]
data_std = []


def normalize(data, train_split):
    global data_mean, data_std
    data_mean = data[:].mean(axis=0)
    data_std = data[:].std(axis=0)
    return (data - data_mean) / data_std

def recoverdata(data):
    global data_mean, data_std
    return data * data_std + data_mean

"""
We can see from the correlation heatmap, few parameters like Relative Humidity and
Specific Humidity are redundant. Hence we will be using select features, not all.
"""

print(
    "The selected parameters are:",
    ", ".join([titles[i] for i in range(len(titles))]),
)
selected_features = [feature_keys[i] for i in range(len(titles))]
features = hist[selected_features]
features.index = range(len(hist))
features.head()

features = normalize(features.values, train_split)
features = pd.DataFrame(features)
features.head()

train_data = features.loc[0 : train_split - 1]
val_data = features.loc[train_split:]

# print(train_data[0:20])
"""
# Training dataset
The training dataset labels starts from the 792nd observation (720 + 72).
"""

start = past + future
end = start + train_split

x_train = train_data[[i for i in range(12)]].values
y_train = features.iloc[start:end][[3]]

# y = []

# for i in range(len(y_train)):
#     y.append(y_train[i:i+days])
# y_train = y
print(x_train[0:20])
print(y_train[0:20])

sequence_length = int(past / step)

"""
The `timeseries_dataset_from_array` function takes in a sequence of data-points gathered at
equal intervals, along with time series parameters such as length of the
sequences/windows, spacing between two sequence/windows, etc., to produce batches of
sub-timeseries inputs and targets sampled from the main timeseries.
"""

dataset_train = keras.preprocessing.timeseries_dataset_from_array(
    x_train,
    y_train,
    sequence_length=sequence_length,
    sampling_rate=step,
    batch_size=batch_size,
)

#print(list(dataset_train)[0][0][0][0])

"""
## Validation dataset
The validation dataset must not contain the last 792 rows as we won't have label data for
those records, hence 792 must be subtracted from the end of the data.
The validation label dataset must start from 792 after train_split, hence we must add
past + future (792) to label_start.
"""

x_end = len(val_data) - past - future

label_start = train_split + past + future

# print('##############')
# print(data_mean)
# print(data_std)
# # print(x_val[-21:-1])
# # print(y_val[-21:-1])
# dataset_val = keras.preprocessing.timeseries_dataset_from_array(
#     x_val,
#     y_val,
#     sequence_length=sequence_length,
#     sampling_rate=step,
#     batch_size=batch_size,
# )
# #Dataset Structure：幾群[] 幾column[] Batch中幾個[] 幾個序列[] 幾行[]變數
# print(list(dataset_val)[-1][0][-1])
# print('re data')
# print(recoverd(list(dataset_val)[-1][0][-1]))
# print('last data')
for batch in dataset_train:
    inputs, targets = batch
print('%%%%%%%%%%%%%%%%%%')
print(inputs[0:2])
print(targets[0:2])
print('=============')
print(inputs.shape)
print(targets.shape)

# for batch_val in dataset_val:
#     inputs_val, targets_val = batch_val
# print(inputs_val)
# print('re data')
# print(recoverd(inputs_val))

print(list(dataset_train)[0:2])


# print("Input shape:", inputs.numpy().shape)
# print("Target shape:", targets.numpy().shape)

"""
## Training
"""

inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
lstm_out = keras.layers.LSTM(32)(inputs)
#lstm2_out = keras.layers.LSTM(32)(lstm_out)
# #lstm_out=keras.layers.Dense(64)(inputs)
outputs = keras.layers.Dense(1)(lstm_out)

# denseout = tf.keras.layers.Dense(units=64, activation='relu')(inputs)
# densout1 = tf.keras.layers.Dense(units=64, activation='relu')(denseout)
# outputs = tf.keras.layers.Dense(units=1)(densout1)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse",)

model.summary()

"""
We'll use the `ModelCheckpoint` callback to regularly save checkpoints, and
the `EarlyStopping` callback to interrupt training when the validation loss
is not longer improving.
"""

path_checkpoint = "model_checkpoint.h5"
# es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

# modelckpt_callback = keras.callbacks.ModelCheckpoint(
#     monitor="val_loss",
#     filepath=path_checkpoint,
#     verbose=1,
#     save_weights_only=True,
#     save_best_only=True,
# )

history = model.fit(
    dataset_train,
    epochs=epochs,
    # validation_data=dataset_val,
    # callbacks=[es_callback, modelckpt_callback],
    
)

"""
We can visualize the loss with the function below. After one point, the loss stops
decreasing.
"""
# save the model to disk
print("[INFO] serializing network...")
model.save(company+"stock_train_dense", save_format="h5")

def visualize_loss(history, title):
    loss = history.history["loss"]
    #val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    #plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


visualize_loss(history, "Training and Validation Loss")

"""
## Prediction
The trained model above is now able to make predictions for 5 sets of values from
validation set.
"""


def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.show()
    return


# for x, y in dataset_val.take(5):
#     show_plot(
#         [x[0][:, 3].numpy(), y[0].numpy(), model.predict(x)[0]],
#         1,
#         "Single Step Prediction",
#     )

future_price = []
real_price = []
print(np.array(list(dataset_train)).shape)

x = np.array([list(dataset_train)[-1][0][-2]])
y =np.array([list(dataset_train)[-1][1][-2]])
print(x)
real_price.append(y[0]*data_std[3] + data_mean[3])
future_price.append(model.predict(x)[0]*data_std[3] + data_mean[3])

x = np.array([list(dataset_train)[-1][0][-1]])
y =np.array([list(dataset_train)[-1][1][-1]])
print(x)
real_price.append(y[0]*data_std[3] + data_mean[3])
future_price.append(model.predict(x)[0]*data_std[3] + data_mean[3])

print(future_price)
print(real_price)
x = x*data_std+data_mean
#y = y*data_std[3] + data_mean[3]
print('y', y)
print('stock', future_price)

future_x = np.array([x_train[-120:]])
print('future',recoverdata(future_x))
print(future_x.shape)
future_price.append(model.predict(future_x)[0]*data_std[3] + data_mean[3])
print('stock', future_price)

time_steps= list(range(-30, 0))
plt.style.use("ggplot")
plt.figure()
plt.plot(time_steps, x[0][-30:,3], '.-',markersize=10,label="History")
plt.plot(0, real_price[1], 'bx',markersize=10,label="Real")
plt.plot(range(-1,2), future_price, 'go',markersize=10,label="Future")
plt.text(-1,np.round(real_price[0])+0.5,np.round(real_price[0][0],2),fontsize=8)
plt.text(0,np.round(real_price[1])+0.5,np.round(real_price[1][0],2),fontsize=8)
plt.text(-1,np.round(future_price[0])+0.5,np.round(future_price[0][0],2),fontsize=8)
plt.text(0,np.round(future_price[1])+0.5,np.round(future_price[1][0],2),fontsize=8)
plt.text(1,np.round(future_price[2])+0.5,np.round(future_price[2][0],2),fontsize=8)
plt.title(company)
plt.xlabel("Days#")
plt.ylabel("Price")

plt.legend()
t = time.localtime()
num = str(t.tm_year)+str(t.tm_mon)+str(t.tm_mday)+str(t.tm_min)+str(t.tm_sec)
plt.savefig(company+"futuretrain"+num+'.png')

plt.show()