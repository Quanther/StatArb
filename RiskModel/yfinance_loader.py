import yfinance as yf
import pandas as pd

def get_stock_data(tickers, start_date, end_date):
    """
    Fetch adjusted close prices for the given tickers from Yahoo Finance.
    """
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

if __name__ == "__main__":
    
    url_dow = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    dow_table = pd.read_html(url_dow, header=0)[2]
    dow_tickers = dow_table['Symbol'].tolist()

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)[0]
    sp500_tickers = table['Symbol'].tolist()

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Example tickers
    tickers = dow_tickers
    start_date = '2021-01-01'
    end_date = '2025-01-01'
    
    stock_price = get_stock_data(sp500_tickers, start_date, end_date)['Adj Close']
    stock_price = stock_price.dropna(axis=1)
    stock_returns = stock_price.pct_change().dropna()
    print(stock_returns.head())
