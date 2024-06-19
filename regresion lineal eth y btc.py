import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

bitcoin_df = pd.read_csv('BTC-USD (2014-2024).csv')
bitcoin_df['Date'] = pd.to_datetime(bitcoin_df['Date'])
#cambiar a fechas importantes 
target_date = pd.Timestamp('2023-06-12')
start_date = target_date - pd.Timedelta(days=10)
end_date = target_date + pd.Timedelta(days=10)
bitcoin_subset = bitcoin_df[(bitcoin_df['Date'] >= start_date) & (bitcoin_df['Date'] <= end_date)]
X = np.array([(date - bitcoin_subset['Date'].min()).days for date in bitcoin_subset['Date']]).reshape(-1, 1)
y = bitcoin_subset['Open'].values
X_mean = np.mean(X)
y_mean = np.mean(y)
num = np.sum((X - X_mean) * (y - y_mean))
den = np.sum((X - X_mean) ** 2)
slope = num / den
intercept = y_mean - slope * X_mean
y_pred = slope * X + intercept
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y_mean) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print('resultados de regrecion bitcoin:')
print(f'pendiente: {slope}')
print(f'intercepto: {intercept}')
print(f'R cuadrada: {r_squared}')
print("-------------------------")
ethereum_df = pd.read_csv('ETH-USD (2017-2024).csv')
ethereum_df['Date'] = pd.to_datetime(ethereum_df['Date'])
ethereum_subset = ethereum_df[(ethereum_df['Date'] >= start_date) & (ethereum_df['Date'] <= end_date)]
X = np.array([(date - ethereum_subset['Date'].min()).days for date in ethereum_subset['Date']]).reshape(-1, 1)
y = ethereum_subset['Open'].values
X_mean = np.mean(X)
y_mean = np.mean(y)
num = np.sum((X - X_mean) * (y - y_mean))
den = np.sum((X - X_mean) ** 2)
slope = num / den
intercept = y_mean - slope * X_mean
y_pred = slope * X + intercept
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y_mean) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print('resultados de regrecion ethereum:')
print(f'pendiente: {slope}')
print(f'intercepto: {intercept}')
print(f'R cuadrada: {r_squared}')