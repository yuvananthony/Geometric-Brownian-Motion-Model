import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

td = datetime.today()

stock_data = yf.download("^GSPC", start=td - timedelta(days = 365), end = td)
stock_prices = stock_data['Close']

log_returns = np.log(stock_prices / stock_prices.shift(1)).dropna()

vol = log_returns.rolling(window = 21).std().dropna()
S0 = stock_prices.iloc[-1]
delta_t = 1/252
mu = 0.0436
sigma = float(vol.dropna().iloc[-1])

sim_per_day = 10000
sim_days = 252

sims = np.zeros((sim_days + 1, sim_per_day))
sims[0, :] = S0

for t in range(1, sim_days + 1):
    sims[t] = sims[t - 1] * np.exp((mu - (sigma ** 2) / 2) * delta_t + sigma * np.sqrt(delta_t) * np.random.standard_normal(sim_per_day))

for i in range(100):
    plt.plot(sims[:, i], color='blue', alpha=0.1)
plt.title("Forecasted Returns for SP500")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()
