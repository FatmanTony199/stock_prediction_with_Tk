import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplcursors
import tkinter as tk
from tkinter import ttk, messagebox

BAND_OPTIONS = {
    "日K低點買進, 日K高點賣出": {"buy_price": "Low", "sell_price": "High"},
    "日K高點買進, 日K低點賣出": {"buy_price": "High", "sell_price": "Low"},
}

def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError("無法下載股票資料，請確認股票代號或日期。")
    return df

def simulate_trading(df: pd.DataFrame, option: str, capital: float) -> (pd.DataFrame, float):
    prices = BAND_OPTIONS[option]
    df = df.copy()
    df["Buy"] = df[prices["buy_price"]]
    df["Sell"] = df[prices["sell_price"]].shift(-1)
    df.dropna(inplace=True)

    df["Shares"] = (capital / df["Buy"]).astype(int)  # 每次交易買進幾股
    df["Profit"] = (df["Sell"] - df["Buy"]) * df["Shares"]
    df["Cumulative Profit"] = df["Profit"].cumsum()
    df["Remaining Capital"] = capital + df["Cumulative Profit"]
    return df, df["Remaining Capital"].iloc[-1]

def plot_trading_results(df: pd.DataFrame, ticker: str, option: str):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Cumulative Profit"], marker='o', label="累積損益")
    plt.title(f"{ticker} - 策略: {option}")
    plt.xlabel("日期")
    plt.ylabel("損益 (USD)")
    plt.grid(True)
    plt.legend()
    mplcursors.cursor(hover=True)
    plt.tight_layout()
    plt.show()

def on_run():
    ticker = entry_ticker.get().strip().upper()
    strategy = strategy_var.get()
    start_date = entry_start.get()
    end_date = entry_end.get()
    try:
        capital = float(entry_capital.get())
    except ValueError:
        messagebox.showerror("錯誤", "請輸入有效的初始資金（數字）")
        return

    if not ticker or strategy not in BAND_OPTIONS:
        messagebox.showerror("錯誤", "請確認股票代號與策略選擇")
        return

    try:
        df = download_stock_data(ticker, start=start_date, end=end_date)
        result_df, final_capital = simulate_trading(df, strategy, capital)
        plot_trading_results(result_df, ticker, strategy)
        result_text.set(f"模擬完成！交易次數: {len(result_df)}\n最終資金: ${final_capital:,.2f}")
    except Exception as e:
        messagebox.showerror("錯誤", str(e))

# --- Tkinter GUI ---
root = tk.Tk()
root.title("簡易股票交易模擬器")
root.geometry("400x350")

tk.Label(root, text="股票代號（如 AAPL）").pack()
entry_ticker = tk.Entry(root)
entry_ticker.pack()

tk.Label(root, text="開始日期（YYYY-MM-DD）").pack()
entry_start = tk.Entry(root)
entry_start.insert(0, "2022-01-01")
entry_start.pack()

tk.Label(root, text="結束日期（YYYY-MM-DD）").pack()
entry_end = tk.Entry(root)
entry_end.insert(0, "2024-01-01")
entry_end.pack()

tk.Label(root, text="選擇交易策略").pack()
strategy_var = tk.StringVar()
strategy_menu = ttk.Combobox(root, textvariable=strategy_var, values=list(BAND_OPTIONS.keys()))
strategy_menu.current(0)
strategy_menu.pack()

tk.Label(root, text="初始資金（USD）").pack()
entry_capital = tk.Entry(root)
entry_capital.insert(0, "10000")
entry_capital.pack()

tk.Button(root, text="開始模擬", command=on_run).pack(pady=10)

result_text = tk.StringVar()
tk.Label(root, textvariable=result_text, fg="blue").pack()

root.mainloop()
