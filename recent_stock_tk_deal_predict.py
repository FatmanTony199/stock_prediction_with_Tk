import yfinance as yf
import pandas as pd
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
import mplcursors
import tensorflow as tf
from tensorflow import keras
from keras import models  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

com = ["2330.TW"]
rp = None  
future_price = []
real_price = []
data_std = []
data_mean = []

def normalize(data):
    global data_mean, data_std
    data_mean = data[:].mean(axis=0)
    data_std = data[:].std(axis=0)
    return (data - data_mean) / data_std


def train():
    global var_c, var_d, box, data_mean, data_std, var_ep, sequence_length, dataset_train,inputs, var_ba
    split_fraction = 1
    past = 120
    future = 0
    step = 1
    train_data =[]
    ticker_symbol = var_c.get() if var_c.get() in com else "2330.TW"
    days = int(var_d.get())
    # data = int(var_dt.get())
    msft = yf.Ticker(ticker_symbol)
    hist = msft.history(period="max")
    train_split = int(hist.shape[0])
    start = past + future
    end = start + train_split

    if hist.empty:
        messagebox.showerror("錯誤", "股票代碼無效或無數據")
        return

    sma20 = hist["Close"].rolling(window=20).mean() 
    std20 = hist["Close"].rolling(window=20).std()
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20   
    hist["upper_band"] = upper_band         
    hist["lower_band"] = lower_band           
    hist["sma20"] = sma20                            
    #hist["Day"] = np.arange(len(hist))  
    hist = hist.fillna(0)

    if box.get() == "高通道":
        point = upper_band
    elif box.get() == "高通一標準差":
        point = sma20 + 1 * std20
    elif box.get() == "20日平均線":
        point = sma20
    elif box.get() == "低通一標準差":
        point = sma20 - 1 * std20
    elif box.get() == "低通道":
        point = lower_band
    else:
        point = sma20

    balance_history = {}
    balance = 100000 
    stock = 0  

    for i in range(len(hist) - days, len(hist)):
        if hist["Close"].iloc[i] < point.iloc[i - 1] and stock == 0:
            stock = balance // hist["Close"].iloc[i]
            balance -= stock * hist["Close"].iloc[i]
        elif hist["Close"].iloc[i] > point.iloc[i - 1] and stock > 0:
            balance += stock * hist["Close"].iloc[i]
            stock = 0

        balance_history[hist.index[i]] = balance + stock * hist["Close"].iloc[i]

    feature_columns = ["Open", "Close", "upper_band", "lower_band", "sma20", "Volume"]
    feature = hist[feature_columns]
    feature.index = range(len(hist))
    

    feature = normalize(feature)
    feature = pd.DataFrame(feature)
    print(feature)
    train_data = feature.loc[0:train_split-1]
    val_data = feature.loc[train_split:]

    print(train_data)
    x_train = train_data
    y_train = feature.pop("Close")

    sequence_length = int(past / step)

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=int(var_ba.get()),
    )
    for batch in dataset_train:
        inputs, targets = batch

    inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
    lstm_out = keras.layers.LSTM(32)(inputs)
    #lstm2_out = keras.layers.LSTM(32)(lstm_out)
    # #lstm_out=keras.layers.Dense(64)(inputs)
    outputs = keras.layers.Dense(1)(lstm_out)

    # denseout = tf.keras.layers.Dense(units=64, activation='relu')(inputs)
    # densout1 = tf.keras.layers.Dense(units=64, activation='relu')(denseout)
    # outputs = tf.keras.layers.Dense(units=1)(densout1)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse",)
    model.summary()
    history = model.fit(
    dataset_train,
    epochs=int(var_ep.get()),

    
)
    model.save(ticker_symbol+"_stock_train_dense", save_format="h5")

def predict():
    global dataset_train, ticker_symbol, data_std, data_mean
    ticker_symbol = var_c.get() if var_c.get() in com else "2330.TW"
    model = models.load_model(ticker_symbol+"_stock_train_dense")
    

    print(np.array(list(dataset_train), dtype="object").shape)

    x = np.array([list(dataset_train)[-1][0][-2]])
    y = np.array([list(dataset_train)[-1][1][-2]])
    print(x)
    real_price.append(y[0]*data_std[3] + data_mean[3])
    future_price.append(model.predict(x)[0]*data_std[3] + data_mean[3])

    x = np.array([list(dataset_train)[-1][0][-1]])
    y = np.array([list(dataset_train)[-1][1][-1]])
    print(x)
    real_price.append(y[0]*data_std[3] + data_mean[3])
    future_price.append(model.predict(x)[0]*data_std[3] + data_mean[3])

    print(future_price)
    print(real_price)
    print("////////////////////")
    print("std:", data_std)
    print("mean", data_mean)
    x = x*data_std+data_mean
    print("y", y)
    print("stock", future_price)

    future_x = np.array([x_train[-120:]])
    print("future", recoverdata(future_x))
    print(future_x.shape)
    future_price.append(model.predict(future_x)[0]*data_std[3] + data_mean[3])
    print("stock", future_price)

    time_steps = list(range(-30, 0))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(time_steps, x[0][-30:,3], '.-', markersize = 10, label = "History")
    plt.plot(0, real_price[1], 'bx', markersize = 10, label = "Real")
    plt.plot(range(-1, 2), future_price, 'go', markersize = 10, label = "Future")
    plt.text(-1, np.round(real_price[0])+0.5, np.round(real_price[0][0], 2), fontsize = 8)
    plt.text(0, np.round(real_price[1])+0.5, np.round(real_price[1][0], 2), fontsize = 8)
    plt.text(-1, np.round(future_price[0])+0.5, np.round(future_price[0][0], 2), fontsize = 8)
    plt.text(0, np.round(future_price[1])+0.5, np.round(future_price[1][0], 2), fontsize = 8)
    plt.text(1, np.round(future_price[2])+0.5, np.round(future_price[2][0], 2), fontsize = 8)
    plt.title(ticker_symbol)
    plt.xlabel("Days#")
    plt.ylabel("Price")

    plt.legend()
    t = time.localtime()
    num = str(t.tm_year)+str(t.tm_mon)+str(t.tm_mday)+str(t.tm_min)+str(t.tm_sec)
    plt.savefig(ticker_symbol+"_future"+num+'.png')

    plt.show()


def main():
    global var_c, var_d, var_dt, box, tk, var_ep, var_ba
    root = tk.Tk()
    root.title("股票預測")

    frame = tk.Frame(root)
    frame.pack(pady=10)
    tk.Label(frame, text="股票代碼:").pack(side=tk.LEFT)
    var_c = tk.StringVar(value="2330.TW")
    entry_c = tk.Entry(frame, textvariable=var_c, width=10)
    entry_c.pack(side=tk.LEFT)

    frame2 = tk.Frame(root)
    frame2.pack(pady=10)
    tk.Label(frame2, text="預測天數:").pack(side=tk.LEFT)
    var_d = tk.StringVar(value="100")
    entry_d = tk.Entry(frame2, textvariable=var_d, width=10)
    entry_d.pack(side=tk.LEFT)

    frame3 = tk.Frame(root)
    frame3.pack(pady=10)
    tk.Label(frame3, text="Epochs:").pack(side=tk.LEFT)
    var_ep = tk.StringVar(value="20")
    entry_ep = tk.Entry(frame3, textvariable=var_ep, width=10)
    entry_ep.pack(side=tk.LEFT)

    frame4 = tk.Frame(root)
    frame4.pack(pady=10)
    tk.Label(frame4, text="參考通道:").pack(side=tk.LEFT)
    box = ttk.Combobox(frame4, values=["高通道", "高通一標準差", "20日平均線", "低通一標準差", "低通道"])
    box.pack(side=tk.LEFT)
    box.set("高通道")

    frame5 = tk.Frame(root)
    frame5.pack(pady=10)
    tk.Label(frame5, text="Batch:").pack(side=tk.LEFT)
    var_ba = tk.StringVar(value="32")
    entry_ba = tk.Entry(frame5, textvariable=var_ba, width=10)
    entry_ba.pack(side=tk.LEFT)


    btn = tk.Button(root, text="開始訓練", command=train)
    btn.pack(padx=10, pady=20)

    btn = tk.Button(root, text="開始預測", command=predict)
    btn.pack(padx=20, pady=20)

    root.mainloop()

if __name__=="__main__":
    main()