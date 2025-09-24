import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
com=["2330.TW"]
rp = 0.0
return_rate = 0
def sk():
	global tk, var_c, var_d, var_dt, box, return_rate, rp, root

	if var_c.get() in com: 
		msft = yf.Ticker(var_c.get())
	else:
		msft = yf.Ticker("2330.TW")

	hist = msft.history(period = "max")
	sma20 = hist["Close"].rolling(window=20).mean()
	std20 = hist["Close"].rolling(window=20).std()
	days = int(var_d.get())
	data = int(var_dt.get())
	upper_band = sma20 + 2 * std20
	lower_band = sma20 - 2 * std20

	if box.get() == "高通道":
		point = sma20 + 2 * std20

	elif box.get() == "高通一標準差":
		point = sma20 + 1 * std20

	elif box.get() == "20日平均線":
		point = sma20 

	elif box.get() == "低通一標準差":
		point = sma20 - 1 * std20

	elif box.get() == "低通道":
		point = sma20 - 2 * std20

	else:
		point = sma20

	balance_history = {}
	balance = 100000
	stock = 0

	for i in range(len(hist) - days, len(hist)):  #buy
		if hist["Close"].iloc[i] < point.iloc[i] and stock == 0 and i - len(hist)+days!=0:
			stock = balance // hist["Close"].iloc[i]
			balance -= stock * hist["Close"].iloc[i]

		elif hist["Close"].iloc[i] > point.iloc[i] and stock > 0 and i -len(hist)+days!=0: #sell
			balance += stock * hist["Close"].iloc[i]
			stock = 0

		balance_history[hist.index[i]] = (balance + stock * hist["Close"].iloc[i])

	final_value = balance + stock * hist["Close"].iloc[-1]
	return_rate = (final_value - 100000) / 100000
	print("最終報酬率: %.2f%%" % (return_rate * 100))
	rp.config(text= str(round(float(return_rate * 100), 2))+"%")

	N = days
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, N), hist["Close"].iloc[len(hist)-days:len(hist)], label="Close")
	plt.plot(np.arange(0, N), upper_band.iloc[len(hist)-days:len(hist)], label="UPPER")
	plt.plot(np.arange(0, N), lower_band.iloc[len(hist)-days:len(hist)], label="LOWER")
	plt.plot(np.arange(0, N), sma20.iloc[len(hist)-days:len(hist)], label="MEAN")

	for i in range(days-data, days):
		j = i + len(hist)-days
		plt.text(i-0.7, round(hist["Close"].iloc[j])+0.5, round(hist["Close"].iloc[j],2),fontsize = 8)

	plt.title(var_c.get())
	plt.xlabel("Days#")
	plt.ylabel("Price")

	plt.legend()
	plt.savefig(var_c.get()+".png")
	plt.show()

#/////////////////////////////////////////////////////////////////////////////////////////////////////////
def main():
	global tk, var_c, var_d, var_dt, box, rp
	width = 1000
	height = 300
	left = 0
	top = 0

	root = tk.Tk()
	root.configure(background='#C8191D') 
	root.title('stock')
	root.geometry(f'{width}x{height}+{left}+{top}')

	# frame = tk.Frame(root, pady=10, padx=10, bg='#09c')

	ca = tk.Label(root, text='Company:')
	ca.grid(row=0, column=1)
	var_c = tk.StringVar()
	myentry_c = tk.Entry(root, textvariable=var_c)
	myentry_c.grid(row=0, column=2)

	d = tk.Label(root, text='Days:')
	d.grid(row=2, column=1)
	var_d = tk.StringVar()
	myentry_d = tk.Entry(root, textvariable=var_d)
	myentry_d.grid(row=2, column=2)

	dt = tk.Label(root, text='Datas:')
	dt.grid(row=4, column=1)
	var_dt = tk.StringVar()
	myentry_dt = tk.Entry(root, textvariable=var_dt)
	myentry_dt.grid(row=4, column=2)

	ch = tk.Label(root, text='低點選擇:')
	ch.grid(row=6, column=1)
	box = ttk.Combobox(root, values=['高通道', '高通一標準差','20日平均線','低通一標準差','低通道'])
	box.grid(row=6, column=2)

	r = tk.Label(root, text='return:')
	r.grid(row=8, column=1)
	rp = tk.Label(root, text= "%.2f%%" % (return_rate * 100))
	rp.grid(row=8, column=2)

	mybutton2 = tk.Button(root, text='enter', command=sk)
	mybutton2.grid(row=10, column=2)

	root.mainloop()

if __name__ =='__main__':
	main()