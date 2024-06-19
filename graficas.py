import pandas as pd
import matplotlib.pyplot as plt

bitcoin_df = pd.read_csv('BTC-USD (2014-2024).csv')
bitcoin_df['Date'] = pd.to_datetime(bitcoin_df['Date'])
bitcoin_df.set_index('Date', inplace=True)
def plot_bitcoin_price(target_date):
    start_date = target_date - pd.Timedelta(days=10)
    end_date = target_date + pd.Timedelta(days=10)
    bitcoin_subset = bitcoin_df.loc[start_date:end_date]
    plt.figure(figsize=(12, 6))
    plt.plot(bitcoin_subset.index, bitcoin_subset['Open'])
    plt.axvline(x=target_date, color='r', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Open Price (USD)')
    plt.title(f'Bitcoin en {target_date.strftime("%Y-%m-%d")}')
    plt.savefig(f'{target_date.strftime("%Y-%m-%d")}.jpg')
    plt.show()
plot_bitcoin_price(pd.Timestamp('2017-12-22'))
plot_bitcoin_price(pd.Timestamp('2022-06-12'))
plot_bitcoin_price(pd.Timestamp('2022-06-13'))
