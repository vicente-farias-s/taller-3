import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

#yo no cree este codigo, tome un esqueleto desde stack, 
#y despues fui pidiendo ayuda a conocidos y gpt 4
#para ir entendiendo el como funciona, pense en agregarlo,
#ya que si bien no lo hice, si entendi el como se hizo, 
#y para que sirve

# Cargar el archivo CSV de Bitcoin
bitcoin_df = pd.read_csv('BTC-USD (2014-2024).csv')

# Convertir la columna 'Date' a formato de fecha
bitcoin_df['Date'] = pd.to_datetime(bitcoin_df['Date'])
bitcoin_df.set_index('Date', inplace=True)

# Seleccionar los 21 días alrededor del 2023-06-12
target_date = pd.Timestamp('2023-06-12')
start_date = target_date - pd.Timedelta(days=10)
end_date = target_date + pd.Timedelta(days=10)
bitcoin_subset = bitcoin_df.loc[start_date:end_date]

# Generar el gráfico
plt.figure(figsize=(12, 6))
plt.plot(bitcoin_subset.index, bitcoin_subset['Open'])
plt.axvline(x=target_date, color='r', linestyle='--', label='2023-06-12')
plt.xlabel('Date')
plt.ylabel('Open Price (USD)')
plt.title('Bitcoin Prices Around 2023-06-12')
plt.legend()
plt.show()

# Función para calcular la autocorrelación parcial
def pacf(x, max_lag):
    n = len(x)
    r = np.zeros(max_lag)
    for k in range(1, max_lag+1):
        numerator = np.sum((x[k:] - np.mean(x[k:])) * (x[:-k] - np.mean(x[:-k])))
        denominator = np.sum((x - np.mean(x))**2)
        r[k-1] = numerator / denominator
    return r

# Ajustar el modelo ARIMA
p = 1
d = 1
q = 1
y = bitcoin_subset['Open']
n = len(y)

# Calcular los coeficientes del modelo ARIMA
phi = np.zeros(p)
theta = np.zeros(q)
sigma2 = 0

for i in range(10):
    # Calcular los nuevos coeficientes
    y_hat = np.zeros(n)
    for t in range(p, n):
        y_hat[t] = phi[0] * y[t-1]
    for t in range(q, n):
        y_hat[t] -= theta[0] * (y[t-1] - y_hat[t-1])
    
    # Calcular el error cuadrático medio
    e = y - y_hat
    sigma2 = np.sum(e**2) / n
    
    # Actualizar los coeficientes
    phi[0] = np.sum((y[p:] - np.mean(y[p:])) * (y[:-p] - np.mean(y[:-p]))) / np.sum((y[:-p] - np.mean(y[:-p]))**2)
    theta[0] = -np.sum(e[q:] * e[:-q]) / np.sum(e[:-q]**2)

print('Bitcoin ARIMA Model Summary:')
print(f'p: {p}, d: {d}, q: {q}')
print(f'Phi: {phi}')
print(f'Theta: {theta}')
print(f'Sigma^2: {sigma2}')

# Hacer predicciones
forecast = np.zeros(10)
for i in range(10):
    forecast[i] = phi[0] * y[-1]
    y = np.append(y, forecast[i])
    y = y[1:]

print('\nBitcoin Forecast:')
print(forecast)

# Cargar el archivo CSV de Ethereum
ethereum_df = pd.read_csv('ETH-USD (2017-2024).csv')

# Convertir la columna 'Date' a formato de fecha
ethereum_df['Date'] = pd.to_datetime(ethereum_df['Date'])
ethereum_df.set_index('Date', inplace=True)

# Seleccionar los 21 días alrededor del 2023-06-12
ethereum_subset = ethereum_df.loc[start_date:end_date]

# Ajustar el modelo ARIMA
p = 1
d = 1
q = 1
y = ethereum_subset['Open']
n = len(y)

# Calcular los coeficientes del modelo ARIMA
phi = np.zeros(p)
theta = np.zeros(q)
sigma2 = 0

for i in range(10):
    # Calcular los nuevos coeficientes
    y_hat = np.zeros(n)
    for t in range(p, n):
        y_hat[t] = phi[0] * y[t-1]
    for t in range(q, n):
        y_hat[t] -= theta[0] * (y[t-1] - y_hat[t-1])
    
    # Calcular el error cuadrático medio
    e = y - y_hat
    sigma2 = np.sum(e**2) / n
    
    # Actualizar los coeficientes
    phi[0] = np.sum((y[p:] - np.mean(y[p:])) * (y[:-p] - np.mean(y[:-p]))) / np.sum((y[:-p] - np.mean(y[:-p]))**2)
    theta[0] = -np.sum(e[q:] * e[:-q]) / np.sum(e[:-q]**2)

print('\nEthereum ARIMA Model Summary:')
print(f'p: {p}, d: {d}, q: {q}')
print(f'Phi: {phi}')
print(f'Theta: {theta}')
print(f'Sigma^2: {sigma2}')

# Hacer predicciones
forecast = np.zeros(10)
for i in range(10):
    forecast[i] = phi[0] * y[-1]
    y = np.append(y, forecast[i])
    y = y[1:]

print('\nEthereum Forecast:')
print(forecast)