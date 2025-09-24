import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
import mplcursors

com=["2330.TW"]
rp = 0.0
return_rate = 0

def sk():
	global tk, var_c, var_d, box_b, return_rate, rp, root, box_s, var_f, var_dt

	if var_c.get() in com: 
		msft = yf.Ticker(var_c.get())
	else:
		msft = yf.Ticker("2330.TW")

	hist = msft.history(period = "max")
	
	if hist.empty:
		messagebox.showerror("錯誤", f"股票代碼「{var_c.get()}」查無資料")
		return

	sma20 = hist["Close"].rolling(window=20).mean()
	std20 = hist["Close"].rolling(window=20).std()
	days = min(int(var_d.get()), len(hist))
	data = int(var_dt.get()) if var_dt.get() != "" else 5

	upper_band = sma20 + 2 * std20
	lower_band = sma20 - 2 * std20

	if box_b.get() == "高通道":
		point_b = upper_band

	elif box_b.get() == "高通一標準差":
		point_b = sma20 + 1 * std20

	elif box_b.get() == "20日平均線":
		point_b = sma20 

	elif box_b.get() == "低通一標準差":
		point_b = sma20 - 1 * std20

	elif box_b.get() == "低通道":
		point_b = lower_band

	else:
		point_b = sma20
#//////////////////////////////////////////////////
	if box_s.get() == "高通道":
		point_s = upper_band

	elif box_s.get() == "高通一標準差":
		point_s = sma20 + 1 * std20

	elif box_s.get() == "20日平均線":
		point_s = sma20 

	elif box_s.get() == "低通一標準差":
		point_s = sma20 - 1 * std20

	elif box_s.get() == "低通道":
		point_s = lower_band

	else:
		point_s = sma20

	balance_history = {}
	balance = ori_balance = int(var_f.get()) if var_f.get() != "" else 100000
	stock = 0

	for i in range(len(hist) - days, len(hist)):  #buy
		if hist["Close"].iloc[i] < point_b.iloc[i] and stock == 0 and i - len(hist)+days!=0:
			stock = balance // hist["Close"].iloc[i]
			balance -= stock * hist["Close"].iloc[i]

		elif hist["Close"].iloc[i] > point_s.iloc[i-1] and stock > 0 and i -len(hist)+days!=0: #sell
			balance += stock * hist["Close"].iloc[i]
			stock = 0


	balance_history[hist.index[i]] = (balance + stock * hist["Close"].iloc[i])
	print(balance_history)
	final_value = balance + stock * hist["Close"].iloc[-1]
	return_rate = (final_value - ori_balance) / ori_balance
	rp.config(text=f"{round(return_rate * 100, 2)}%")

	N = days
	fig, ax1 = plt.subplots(figsize=(12, 8)) 
	plt.style.use("ggplot")

	price_line, = ax1.plot(np.arange(0, N), hist["Close"].iloc[len(hist) - days:], 
						   label="Close Price", linewidth=2, color="blue")
	ax1.plot(np.arange(0, N), upper_band.iloc[len(hist) - days:], 
			 label="Upper Band", linestyle="--", color="red")
	ax1.plot(np.arange(0, N), lower_band.iloc[len(hist) - days:], 
			 label="Lower Band", linestyle="--", color="green")
	ax1.plot(np.arange(0, N), sma20.iloc[len(hist) - days:], 
			 label="20-Day SMA", linestyle="-.", color="orange")

	for i in range(days - data, days):
		j = i + len(hist) - days
		ax1.text(i, hist["Close"].iloc[j] + 0.5, round(hist["Close"].iloc[j], 2), 
				 fontsize=8, color="black", bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

	ax1.set_title(var_c.get(), fontsize=16, fontweight="bold")
	ax1.set_ylabel("Price", fontsize=12)
	ax1.legend(fontsize=10, loc="upper left")
	ax1.grid(alpha=0.6)

	ax2 = ax1.twinx()  
	volume_line, = ax2.plot(np.arange(0, N), hist["Volume"].iloc[len(hist) - days:] / 1e6, 
							label="Volume (1 mil)", linewidth=2, color="purple", alpha=0.6)

	for i in range(days - data, days):
		j = i + len(hist) - days
		ax2.text(i, hist["Volume"].iloc[j] / 1e6 + 0.05, round(hist["Volume"].iloc[j] / 1e6, 2), 
				 fontsize=8, color="black", bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

	ax2.set_ylabel("Volume (1 mil)", fontsize=12)
	ax2.legend(fontsize=10, loc="upper right")

	cursor = mplcursors.cursor([price_line, volume_line], hover=True)
	cursor.connect("add", lambda sel: sel.annotation.set_text(
		f"Day: {int(sel.index)}\nPrice: {sel.target[1]:.2f}" if sel.artist == price_line else f"Day: {int(sel.index)}\nVolume: {sel.target[1]:.2f}"
	))

	plt.tight_layout()
	plt.savefig(var_c.get() + "_combined.png", dpi=300) if var_c.get() != "" else plt.savefig("2330.TW_combined.png", dpi=300)
	plt.show()

#/////////////////////////////////////////////////////////////////////////////////////////////////////////
def main():
	global tk, var_c, var_d, box_b, rp, box_s, var_f, var_dt
	width = 400
	height = 300
	left = 0
	top = 0

	root = tk.Tk()
	root.configure(background='#2C2C2C')  
	root.title('Self-Destructing Book')
	root.geometry(f'{width}x{height}+{left}+{top}')

	ca = tk.Label(root, text='Company:', bg='#2C2C2C', fg='white', font=('Arial', 12)) 
	ca.grid(row=0, column=1, padx=10, pady=5, sticky='e')
	var_c = tk.StringVar()
	myentry_c = tk.Entry(root, textvariable=var_c, font=('Arial', 12), bg='#3E3E3E', fg='white', insertbackground='white')
	myentry_c.grid(row=0, column=2, padx=10, pady=5, sticky='w')

	d = tk.Label(root, text='Days:', bg='#2C2C2C', fg='white', font=('Arial', 12))
	d.grid(row=2, column=1, padx=10, pady=5, sticky='e')
	var_d = tk.StringVar()
	myentry_d = tk.Entry(root, textvariable=var_d, font=('Arial', 12), bg='#3E3E3E', fg='white', insertbackground='white')
	myentry_d.grid(row=2, column=2, padx=10, pady=5, sticky='w')

	dt = tk.Label(root, text='Datas:', bg='#2C2C2C', fg='white', font=('Arial', 12))
	dt.grid(row=4, column=1, padx=10, pady=5, sticky='e')
	var_dt = tk.StringVar()
	myentry_dt = tk.Entry(root, textvariable=var_dt, font=('Arial', 12), bg='#3E3E3E', fg='white', insertbackground='white')
	myentry_dt.grid(row=4, column=2, padx=10, pady=5, sticky='w')

	ch_b = tk.Label(root, text='篩選選擇(買):', bg='#2C2C2C', fg='white', font=('Arial', 12))
	ch_b.grid(row=6, column=1, padx=10, pady=5, sticky='e')
	box_b = ttk.Combobox(root, values=['高通道', '高通-標準差', '20日平均線', '低通-標準差', '低通道'], font=('Arial', 12))
	box_b.grid(row=6, column=2, padx=10, pady=5, sticky='w')

	ch_s = tk.Label(root, text='篩選選擇(賣):', bg='#2C2C2C', fg='white', font=('Arial', 12))
	ch_s.grid(row=8, column=1, padx=10, pady=5, sticky='e')
	box_s = ttk.Combobox(root, values=['高通道', '高通-標準差', '20日平均線', '低通-標準差', '低通道'], font=('Arial', 12))
	box_s.grid(row=8, column=2, padx=10, pady=5, sticky='w')

	f = tk.Label(root, text='property:', bg='#2C2C2C', fg='white', font=('Arial', 12))
	f.grid(row=10, column=1, padx=10, pady=5, sticky='e')
	var_f = tk.StringVar()
	myentry_f = tk.Entry(root, textvariable=var_f, font=('Arial', 12), bg='#3E3E3E', fg='white', insertbackground='white')
	myentry_f.grid(row=10, column=2, padx=10, pady=5, sticky='w')

	r = tk.Label(root, text='Return:', bg='#2C2C2C', fg='white', font=('Arial', 12))
	r.grid(row=12, column=1, padx=10, pady=5, sticky='e')
	rp = tk.Label(root, text="0.00%", bg='#2C2C2C', fg='white', font=('Arial', 12, 'bold'))
	rp.grid(row=12, column=2, padx=10, pady=5, sticky='w')

	mybutton2 = tk.Button(root, text='Enter', font=('Arial', 12), bg='#4CAF50', fg='white', activebackground='#45A049', command=sk)
	mybutton2.grid(row=14, column=2, padx=10, pady=15, sticky='w')

	root.mainloop()

if __name__ =='__main__':
	main()