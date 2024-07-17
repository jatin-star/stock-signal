import pandas as pd
from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta as ta
import time
import pytz
from io import BytesIO
import base64
import requests


def index(request):
    pd.DataFrame(columns = ['stock_name', 'score', 'price'])
    return render(request, "index.html")


def datetotimestamp(date):
    time_tuple = date.timetuple()
    timestamp = round(time.mktime(time_tuple))
    return timestamp

def timestamptodate(timestamp):
    return datetime.fromtimestamp(timestamp)

def data_fetch_money_control(symbol):
    start = datetotimestamp(datetime(2024,1,1))
    end = datetotimestamp(datetime.today())
    url = f'https://priceapi.moneycontrol.com/techCharts/indianMarket/stock/history?symbol={symbol}&resolution=1d&from={start}&to={end}&countback=1000&currencyCode=INR'
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        'Accept': '*/*', 'Accept-Encoding': 'gzip, deflate, br', 'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8'
    }

    resp = requests.get(url, headers = header).json()
    data = pd.DataFrame(resp)
    date = []

    for dt in data['t']:
        date.append({'Date':timestamptodate(dt)})

    dt = pd.DataFrame(date)
    intraday_data = pd.concat([dt, data['o'], data['h'], data['l'], data['c'], data['v']], axis = 1)
    intraday_data = intraday_data.rename(columns={'o': 'Open', 'c': 'Close', 'h': 'High', 'l': 'Low', 'v': 'Volume'})
    return intraday_data.set_index('Date')

ist = pytz.timezone('Asia/Kolkata')
def get_current_time():
    current_time_utc = datetime.utcnow()
    current_time_ist = current_time_utc.replace(tzinfo=pytz.utc).astimezone(ist)
    return current_time_ist.strftime("%H:%M:%S")

from datetime import datetime, timedelta

def data_fetch(stock_name):
  try:
      df = data_fetch_money_control(stock_name)
      df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)
  except Exception as e:
      print(f"Error downloading data: {e}")

  return df


def with_itrend(df):
  data = df.copy()

  BBperiod = 21
  BBdeviations = 1.0
  ATRperiod = 5
  i_lookback = 8
  i_atrPeriod = 10
  UseATRfilter = True

  # Calculate Bollinger Bands
  data['SMA'] = ta.sma(data['close'], length=BBperiod)
  data['STD'] = ta.stdev(data['close'], length=BBperiod)
  data['BBUpper'] = data['SMA'] + data['STD'] * BBdeviations
  data['BBLower'] = data['SMA'] - data['STD'] * BBdeviations

  # Initialize trend line and signal columns
  data['TrendLine'] = 0.0
  data['iTrend'] = 0.0
  data['BBSignal'] = np.where(data['close'] > data['BBUpper'], 1, np.where(data['close'] < data['BBLower'], -1, 0))

  # ATR calculation
  data['atr'] = ta.atr(data['high'], data['low'], data['close'], length=ATRperiod)

  for i in range(1, len(data)):
      # Calculate TrendLine
      if data['BBSignal'].iloc[i] == 1 and UseATRfilter:
          data['TrendLine'].iloc[i] = data['low'].iloc[i-1] - data['atr'].iloc[i-1]
          if data['TrendLine'].iloc[i] < data['TrendLine'].iloc[i-1]:
              data['TrendLine'].iloc[i] = data['TrendLine'].iloc[i-1]
      elif data['BBSignal'].iloc[i] == -1 and UseATRfilter:
          data['TrendLine'].iloc[i] = data['high'].iloc[i] + data['atr'].iloc[i]
          if data['TrendLine'].iloc[i] > data['TrendLine'].iloc[i-1]:
              data['TrendLine'].iloc[i] = data['TrendLine'].iloc[i-1]
      elif data['BBSignal'].iloc[i] == 0 and UseATRfilter:
          data['TrendLine'].iloc[i] = data['TrendLine'].iloc[i-1]
      elif data['BBSignal'].iloc[i] == 1 and not UseATRfilter:
          data['TrendLine'].iloc[i] = data['low'].iloc[i]
          if data['TrendLine'].iloc[i] < data['TrendLine'].iloc[i-1]:
              data['TrendLine'].iloc[i] = data['TrendLine'].iloc[i-1]
      elif data['BBSignal'].iloc[i] == -1 and not UseATRfilter:
          data['TrendLine'].iloc[i] = data['high'].iloc[i]
          if data['TrendLine'].iloc[i] > data['TrendLine'].iloc[i-1]:
              data['TrendLine'].iloc[i] = data['TrendLine'].iloc[i-1]
      elif data['BBSignal'].iloc[i] == 0 and not UseATRfilter:
          data['TrendLine'].iloc[i] = data['TrendLine'].iloc[i-1]

      data['iTrend'].iloc[i] = data['iTrend'].iloc[i-1]

      if data['TrendLine'].iloc[i] > data['TrendLine'].iloc[i - 1]:
          data['iTrend'].iloc[i] = 1
      elif data['TrendLine'].iloc[i] < data['TrendLine'].iloc[i - 1]:
          data['iTrend'].iloc[i] = -1

  data['crossupSignal'] = (data['iTrend'].shift(1) < 0) & (data['iTrend'] >= 0)
  data['crossdnSignal'] = (data['iTrend'].shift(1) > 0) & (data['iTrend'] <= 0)

  # Plotting and strategy entries (example with Matplotlib)
  # plt.figure(figsize=(14, 5))
  # plt.plot(data['close'], label='Close Price', alpha=0.5)
  # plt.scatter(data.index[data['crossupSignal']], data['close'][data['crossupSignal']], label='Long Entry', marker='^', color='green')
  # plt.scatter(data.index[data['crossdnSignal']], data['close'][data['crossdnSignal']], label='Short Entry', marker='v', color='red')
  # plt.legend()
  # plt.show()
  return data


def find_first_one(df):
    for idx in range(len(df) - 1, -1, -1):
        if df.iloc[idx]['crossupSignal'] == 1:
            return df.index[idx], 'crossupSignal', df.iloc[idx]['close'], df.iloc[-1]['close']
        if df.iloc[idx]['crossdnSignal'] == 1:
            return df.index[idx], 'crossdnSignal', df.iloc[idx]['close'], df.iloc[-1]['close']
    return None, None

def scores(t_data):
  df = t_data.copy()

  transaction = []
  price = 0

  for i in range(len(df)):
    if df['crossupSignal'][i] == True:
      price = df['close'][i]

    if df['crossdnSignal'][i] == True and price!=0:
      transaction.append(df['close'][i]-price)
      price = 0

  transaction_np = np.array(transaction)
  positive_sum = transaction_np[transaction_np > 0].sum()
  negative_sum = transaction_np[transaction_np < 0].sum()

  profit_factor = ((positive_sum-negative_sum)/positive_sum)
  num_positive_np = np.sum(transaction_np > 0)

  return (num_positive_np/len(transaction_np), profit_factor)


def books(request):
    stock_to_purchase = pd.DataFrame(
        columns=['Date', 'name', 'buy_price', 'current_price', 'profit_factor', 'percentage_profit'])
    stock_to_sell = pd.DataFrame(
        columns=['Date', 'name', 'sell_price', 'current_price', 'profit_factor', 'percentage_profit'])

    stock_list = ['INFY', 'TATAPOWER', 'TEXRAIL', 'WIPRO', 'COALINDIA', 'HDFCBANK', 'ZOMATO']
    # stock_list = ['INFY', 'TATAPOWER', 'TEXRAIL']

    try:
        for stock_name in stock_list:
            data = data_fetch(stock_name)
            t_data = with_itrend(data)
            index, signal, price, current_price = find_first_one(t_data)
            percentage_profit, profit_factor = scores(t_data)
            if index is not None:
                if signal == 'crossupSignal':
                    stock_to_purchase = pd.concat([stock_to_purchase, pd.DataFrame(
                        {'Date': [index], 'name': [stock_name], 'buy_price': price, 'current_price': current_price,
                        'profit_factor': profit_factor, 'percentage_profit': percentage_profit})], ignore_index=True)
                if signal == 'crossdnSignal':
                    stock_to_sell = pd.concat([stock_to_sell, pd.DataFrame(
                        {'Date': [index], 'name': [stock_name], 'sell_price': price, 'current_price': current_price,
                        'profit_factor': profit_factor, 'percentage_profit': percentage_profit})], ignore_index=True)

        # Convert the DataFrames to HTML tables
        buy_df_html = stock_to_purchase.to_html(classes='dataframe table table-bordered', index=False)
        sell_df_html = stock_to_sell.to_html(classes='dataframe table table-bordered', index=False)

        return render(request, "books.html", {"buy_df_html": buy_df_html, "sell_df_html": sell_df_html})
    except Exception as e:
        error_message = f"Connection error or Stock name  incorrect"
        return render(request, "books.html", {"error_message": error_message})


def get_stock_analytics(request):
    stock_analysis_data = pd.DataFrame(
        columns=['Date', 'name', 'buy_price', 'current_price', 'profit_factor', 'percentage_profit']
    )

    if request.method == 'POST':
        stock_name = request.POST.get('stock_name')

        try:
            data = data_fetch(stock_name.upper())
            t_data = with_itrend(data)
            index, signal, price, current_price = find_first_one(t_data)
            percentage_profit, profit_factor = scores(t_data)

            stock_analysis_data = pd.concat([stock_analysis_data, pd.DataFrame(
                {'Date': [index.date()], 'name': [stock_name], 'buy_price': [price], 'current_price': [current_price],
                 'profit_factor': [profit_factor], 'percentage_profit': [percentage_profit]}
            )], ignore_index=True)

            # Plotting
            plt.figure(figsize=(14, 5))
            plt.plot(t_data['close'], label='Close Price', alpha=0.5)
            plt.scatter(t_data.index[t_data['crossupSignal']], t_data['close'][t_data['crossupSignal']], label='Long Entry', marker='^', color='green')
            plt.scatter(t_data.index[t_data['crossdnSignal']], t_data['close'][t_data['crossdnSignal']], label='Short Entry', marker='v', color='red')
            plt.legend()

            # Save plot to a bytes buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()

            # Convert DataFrame to HTML
            df_html = stock_analysis_data.to_html(classes='dataframe table table-bordered',
                                                  index=False) if not stock_analysis_data.empty else "No data found for the given stock name."

            return render(request, "analytics.html", {"df_html": df_html, "stock_name": stock_name, "plot_image": image_base64})
        except Exception as e:
            error_message = f"Error processing stock name '{stock_name}': {str(e)}"
            return render(request, "analytics.html", {"error_message": error_message, "stock_name": stock_name})

    return render(request, "analytics.html")



def analytics(request):
    return render(request, "analytics.html")


def about(request):
    return render(request, "about.html")

