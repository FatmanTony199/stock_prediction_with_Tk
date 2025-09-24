import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# 用 yfinance 讀取股價資料
def load_stock_data(ticker):
	try:
		df = yf.download(ticker, period="1y", interval="1d")
		if df.empty:
			raise ValueError("資料為空")
		df.reset_index(inplace=True)
		# 確保欄位名稱一致
		df = df.rename(columns={"Open":"Open", "High":"High", "Low":"Low", "Close":"Close", "Volume":"Volume"})
		# 選擇必要欄位
		df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
		return df
	except Exception as e:
		messagebox.showerror("資料載入錯誤", f"載入 {ticker} 資料失敗:\n{e}")
		return None

# 特徵標準化
def standardize(df, feature_cols):
	means = df[feature_cols].mean()
	stds = df[feature_cols].std()
	df_scaled = (df[feature_cols] - means) / stds
	return df_scaled, means, stds

# 建立 LSTM 資料集
def create_dataset(data, target_idx, time_steps=20):
	X, y = [], []
	for i in range(len(data) - time_steps):
		X.append(data[i:i+time_steps].values)
		y.append(data.iloc[i+time_steps, target_idx])
	return np.array(X), np.array(y)

COLUMN_MAP = {"Open":0, "High":1, "Low":2, "Close":3, "Volume":4}

class StockPredictorApp:
	def __init__(self, master):
		self.master = master
		master.title("股票多變數LSTM預測")
		master.geometry("480x380")

		# 股票範例，yfinance 支援的代號，TW 要加後綴 .TW
		self.tickers = ["2330.TW", "AAPL", "GOOG", "MSFT"]
		self.feature_cols = ["Open", "High", "Low", "Close", "Volume"]

		self.create_widgets()

		self.data = None
		self.scaled_data = None
		self.means = None
		self.stds = None
		self.model = None
		self.time_steps = 20
		self.batch_size = 16
		self.epochs = 30

	def create_widgets(self):
		frm_top = tk.Frame(self.master)
		frm_top.pack(pady=10)

		tk.Label(frm_top, text="選擇股票代號：").pack(side=tk.LEFT)
		self.var_ticker = tk.StringVar(value=self.tickers[0])
		self.cmb_ticker = ttk.Combobox(frm_top, values=self.tickers, state="readonly", textvariable=self.var_ticker)
		self.cmb_ticker.pack(side=tk.LEFT, padx=5)

		frm_target = tk.Frame(self.master)
		frm_target.pack(pady=10)
		tk.Label(frm_target, text="預測欄位：").pack(side=tk.LEFT)
		self.var_target = tk.StringVar(value="Close")
		self.cmb_target = ttk.Combobox(frm_target, values=list(COLUMN_MAP.keys())[:-1], state="readonly", textvariable=self.var_target)
		self.cmb_target.pack(side=tk.LEFT, padx=5)

		frm_buttons = tk.Frame(self.master)
		frm_buttons.pack(pady=15)

		self.btn_load = tk.Button(frm_buttons, text="載入資料", width=12, command=self.load_data)
		self.btn_load.grid(row=0, column=0, padx=5)

		self.btn_train = tk.Button(frm_buttons, text="訓練模型", width=12, command=self.train_model)
		self.btn_train.grid(row=0, column=1, padx=5)

		self.btn_predict = tk.Button(frm_buttons, text="多步預測", width=12, command=self.multi_step_predict)
		self.btn_predict.grid(row=0, column=2, padx=5)

		self.txt_status = tk.Text(self.master, height=8, width=60)
		self.txt_status.pack(pady=10)

	def log(self, msg):
		self.txt_status.insert(tk.END, msg + "\n")
		self.txt_status.see(tk.END)
		self.master.update()

	def load_data(self):
		ticker = self.var_ticker.get()
		self.log(f"開始從 yfinance 載入 {ticker} 資料...")
		df = load_stock_data(ticker)
		if df is None:
			self.log("資料載入失敗")
			return
		self.data = df
		self.log(f"資料載入完成，共 {len(self.data)} 筆")

		# 標準化
		self.scaled_data, self.means, self.stds = standardize(self.data, self.feature_cols)
		self.log("資料已標準化")

	def train_model(self):
		if self.scaled_data is None:
			messagebox.showwarning("警告", "請先載入資料")
			return

		target = self.var_target.get()
		target_idx = COLUMN_MAP[target]
		self.log(f"開始建立訓練資料，預測欄位: {target}")

		X, y = create_dataset(self.scaled_data, target_idx, self.time_steps)
		if len(X) == 0:
			self.log("資料太少，無法建立訓練集")
			return
		self.log(f"訓練資料建立完成，X形狀: {X.shape}, y形狀: {y.shape}")

		self.model = Sequential()
		self.model.add(LSTM(50, input_shape=(self.time_steps, len(self.feature_cols))))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(1))
		self.model.compile(optimizer="adam", loss="mse")
		self.log("模型建立完成，開始訓練...")

		early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)
		history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=[early_stopping])
		self.log(f"訓練完成，最終損失：{history.history['loss'][-1]:.6f}")

		model_name = f"{self.var_ticker.get()}_predict_{target}.h5"
		self.model.save(model_name)
		self.log(f"模型已儲存為 {model_name}")

	def multi_step_predict(self):
		if self.model is None:
			target = self.var_target.get()
			model_name = f"{self.var_ticker.get()}_predict_{target}.h5"
			if not os.path.exists(model_name):
				messagebox.showwarning("警告", "找不到模型，請先訓練模型")
				return
			self.log(f"載入模型 {model_name}...")
			self.model = load_model(model_name)

		future_steps = 5
		target = self.var_target.get()
		target_idx = COLUMN_MAP[target]

		self.log(f"開始多步預測 {future_steps} 天，預測欄位: {target}")

		last_sequence = self.scaled_data.tail(self.time_steps).values
		predictions = []

		for i in range(future_steps):
			input_seq = np.expand_dims(last_sequence, axis=0)
			pred = self.model.predict(input_seq, verbose=0)[0][0]
			predictions.append(pred)

			next_step = last_sequence[-1].copy()
			next_step[target_idx] = pred
			last_sequence = np.vstack([last_sequence[1:], next_step])

		std_val = float(self.stds[target])
		mean_val = float(self.means[target])
		pred_real = np.array(predictions) * std_val + mean_val

		plt.figure(figsize=(8,4))
		plt.plot(range(1, future_steps+1), pred_real, 'ro-', label=f"Predicted {target}")
		plt.title(f"{self.var_ticker.get()} 多步預測({future_steps} 天) - {target}")
		plt.xlabel("天數 (步數)")
		plt.ylabel(f"{target} 價格")
		plt.legend()
		plt.grid(True)
		plt.show()

		self.log("多步預測完成")


if __name__ == "__main__":
	root = tk.Tk()
	app = StockPredictorApp(root)
	root.mainloop()
