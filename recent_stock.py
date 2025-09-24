import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

company = "2330.TW"
days = 90
msft = yf.Ticker(company)
hist = msft.history(period = "max")

sma20 = hist["Close"].rolling(window=20).mean()
std20 = hist["Close"].rolling(window=20).std()
upper_band = sma20 + 2 * std20
lower_band = sma20 - 2 * std20

balance_history = {}
balance = 100000
stock = 0
for i in range(len(hist) - days, len(hist)):  #buy
	if hist["Close"].iloc[i] < lower_band.iloc[i] and stock == 0:
		stock = balance // hist["Close"].iloc[i]
		balance -= stock * hist["Close"].iloc[i]

	elif hist["Close"].iloc[i] > upper_band.iloc[i] and stock > 0: #sell
		balance += stock * hist["Close"].iloc[i]
		stock = 0

	balance_history[hist.index[i]] = (balance + stock * hist["Close"].iloc[i])

final_value = balance + stock * hist["Close"].iloc[-1]
return_rate = (final_value - 100000) / 100000
print("最終報酬率: %.2f%%" % (return_rate * 100))

N = days
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), hist["Close"].iloc[len(hist)-days:len(hist)], label="Close")
plt.plot(np.arange(0, N), upper_band.iloc[len(hist)-days:len(hist)], label="UPPER")
plt.plot(np.arange(0, N), lower_band.iloc[len(hist)-days:len(hist)], label="LOWER")
plt.plot(np.arange(0, N), sma20.iloc[len(hist)-days:len(hist)], label="MEAN")

for i in range(days):
	j = i + len(hist)-days
	plt.text(i-0.7, round(hist["Close"].iloc[j])+0.5, round(hist["Close"].iloc[j],2),fontsize = 8)

plt.title(company)
plt.xlabel("Days#")
plt.ylabel("Price")

plt.legend()
plt.savefig(company+".png")
plt.show()
