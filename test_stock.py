import yfinance as yf
# msft = yf.Ticker("MSFT").history(period="max")

# if msft.empty:
#     print("error")
# else:
#     print(msft.head()) 
#?//////////////////////////////////////////////////////////////////
ticker = '2330.TW'
data = yf.download(ticker, period='1y')

print(data.index.tz)