import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import mplcursors
import os

# 用 yfinance 讀取股價資料
def load_stock_data(ticker):
	try:
		df = yf.download(ticker, period="1y", interval="1d")
		if df.empty:
			raise ValueError("資料為空")
		df.reset_index(inplace=True)
		df = df.rename(columns={"Open":"Open", "High":"High", "Low":"Low", "Close":"Close", "Volume":"Volume"})
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
		master.title("Product Analytics Management System Dashboard")
		master.geometry("600x450")
		master.configure(bg="#121212")

		# 自訂暗黑主題
		self.style = ttk.Style(master)
		self.set_dark_theme()

		# 股票代號
		self.tickers = ["2330.TW", "AAPL", "GOOG", "MSFT"]
		self.feature_cols = ["Open", "High", "Low", "Close", "Volume"]

		self.data = None
		self.scaled_data = None
		self.means = None
		self.stds = None
		self.model = None
		self.time_steps = 20
		self.batch_size = 16
		self.epochs = 30

		self.create_widgets()

	def set_dark_theme(self):
		# 基本色調
		bg = "#121212"
		fg = "#E0E0E0"
		accent = "#BB86FC"
		btn_bg = "#1F1B24"
		btn_fg = fg
		entry_bg = "#1E1E1E"
		entry_fg = fg

		self.style.theme_use('clam')

		self.style.configure("TLabel", background=bg, foreground=fg, font=("Segoe UI", 10))
		self.style.configure("TFrame", background=bg)
		self.style.configure("TButton", background=btn_bg, foreground=btn_fg, font=("Segoe UI", 10, "bold"), borderwidth=0)
		self.style.map("TButton",
			background=[('active', accent), ('pressed', '#3700B3')],
			foreground=[('active', 'white'), ('pressed', 'white')])
		self.style.configure("TCombobox",
			fieldbackground=entry_bg, background=entry_bg, foreground=entry_fg, font=("Segoe UI", 10))
		self.style.map('TCombobox',
			fieldbackground=[('readonly', entry_bg)],
			background=[('readonly', entry_bg)],
			foreground=[('readonly', entry_fg)])

		self.style.configure("Vertical.TScrollbar",
			background=bg,
			troughcolor=btn_bg,
			bordercolor=bg,
			arrowcolor=fg)

	def create_widgets(self):
		# 標題欄
		title_frame = ttk.Frame(self.master)
		title_frame.pack(fill=tk.X, padx=20, pady=(20,10))

		title_label = ttk.Label(title_frame, text="Product Analytics Management System", font=("Segoe UI", 16, "bold"), foreground="#BB86FC")
		title_label.pack(side=tk.LEFT)

		subtitle_label = ttk.Label(title_frame, text="Dashboard Dark Version by Kostia Varhatiuk for Fireart Studio", font=("Segoe UI", 9), foreground="#666666")
		subtitle_label.pack(side=tk.LEFT, padx=10)

		# 股票選擇區
		selection_frame = ttk.Frame(self.master)
		selection_frame.pack(fill=tk.X, padx=20, pady=10)

		ttk.Label(selection_frame, text="選擇股票代號：").grid(row=0, column=0, sticky=tk.W, pady=5)
		self.var_ticker = tk.StringVar(value=self.tickers[0])
		self.cmb_ticker = ttk.Combobox(selection_frame, values=self.tickers, state="readonly", textvariable=self.var_ticker, width=12)
		self.cmb_ticker.grid(row=0, column=1, sticky=tk.W, padx=5)

		ttk.Label(selection_frame, text="預測欄位：").grid(row=0, column=2, sticky=tk.W, pady=5, padx=(30,0))
		self.var_target = tk.StringVar(value="Close")
		self.cmb_target = ttk.Combobox(selection_frame, values=list(COLUMN_MAP.keys())[:-1], state="readonly", textvariable=self.var_target, width=12)
		self.cmb_target.grid(row=0, column=3, sticky=tk.W, padx=5)

		# 按鈕群組
		button_frame = ttk.Frame(self.master)
		button_frame.pack(fill=tk.X, padx=20, pady=15)

		self.btn_load = ttk.Button(button_frame, text="載入資料", command=self.load_data)
		self.btn_load.grid(row=0, column=0, padx=10)

		self.btn_train = ttk.Button(button_frame, text="訓練模型", command=self.train_model)
		self.btn_train.grid(row=0, column=1, padx=10)

		self.btn_predict = ttk.Button(button_frame, text="多步預測", command=self.multi_step_predict)
		self.btn_predict.grid(row=0, column=2, padx=10)

		# 狀態輸出區 (用tk.Text包在frame內，自訂背景字體顏色)
		status_frame = ttk.Frame(self.master)
		status_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0,20))

		self.txt_status = tk.Text(status_frame, height=10, bg="#1E1E1E", fg="#CCCCCC", insertbackground="#FFFFFF", font=("Consolas", 10), relief=tk.FLAT)
		self.txt_status.pack(fill=tk.BOTH, expand=True)
		self.txt_status.configure(state=tk.DISABLED)

	def log(self, msg):
		self.txt_status.configure(state=tk.NORMAL)
		self.txt_status.insert(tk.END, msg + "\n")
		self.txt_status.see(tk.END)
		self.txt_status.configure(state=tk.DISABLED)
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
			pred = self.model.predict(input_seq)[0][0]
			predictions.append(pred)

			# 移除最舊資料，加上預測結果 (用同一特徵欄位位置填入預測)
			new_row = last_sequence[-1].copy()
			new_row[target_idx] = pred
			last_sequence = np.vstack([last_sequence[1:], new_row])

		# 對預測結果反標準化
		pred_unscaled = np.array(predictions) * self.stds[target] + self.means[target]
		self.log(f"預測結果（反標準化）：\n{pred_unscaled}")

		# 畫圖
		self.plot_results(pred_unscaled, target)

	def plot_results(self, pred_values, target):
		plt.style.use('dark_background')
		fig, ax = plt.subplots(figsize=(10, 5))
		actual = self.data[target].tail(50).values
		ax.plot(range(len(actual)), actual, label="Actual", color="#BB86FC")
		ax.plot(range(len(actual), len(actual) + len(pred_values)), pred_values, label="Predicted", linestyle="--", color="#03DAC6")

		ax.set_title(f"{self.var_ticker.get()} Stock {target} Price Prediction")
		ax.set_xlabel("Days")
		ax.set_ylabel(target)
		ax.legend()
		ax.grid(True, color="#444444")

		# mplcursors 互動顯示
		cursor = mplcursors.cursor(ax.lines, hover=True)
		@cursor.connect("add")
		def on_add(sel):
			x, y = sel.target
			sel.annotation.set(text=f"Day: {int(x)}\nPrice: {y:.2f}", backgroundcolor="#222222", alpha=0.8)

		plt.tight_layout()
		plt.show()

if __name__ == "__main__":
	root = tk.Tk()
	app = StockPredictorApp(root)
	root.mainloop()
