import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_stock_data(tickers, start_date, end_date):
    """
    Fetch adjusted close prices for the given tickers from Yahoo Finance.
    """
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

def get_market_caps(tickers):
    """
    Fetch market capitalizations for given tickers using yfinance.Tickers().
    """
    # Create a Tickers object for batch processing
    ticker_data = yf.Tickers(tickers)
    market_caps = []

    # Extract market caps
    for ticker in tickers:
        try:
            cap = ticker_data.tickers[ticker].info.get('marketCap', 0)
            # data = ticker_data.history(period="1d")
            if cap:  # Skip stocks with missing market cap
                market_caps.append((ticker, cap))
        except Exception as e:
            print(f"Error fetching market cap for {ticker}: {e}")

    # Sort by market cap in descending order
    return sorted(market_caps, key=lambda x: x[1], reverse=True)

def fetch_sp500_companies():
    url = 'https://www.slickcharts.com/sp500'
    
    # Add headers to mimic a real browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Ensure the request was successful

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'class': 'table table-hover table-borderless table-sm'})
    
    if not table:
        raise ValueError("Could not find the table. The page structure may have changed.")

    rows = table.find_all('tr')[1:]  # Skip the header row

    companies = []
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 3:
            # rank = cols[0].text.strip()
            # ticker = cols[1].text.strip()
            company_name = cols[2].text.strip()
            # weight = cols[3].text.strip()
            
            if '.' in company_name: company_name = company_name.replace('.', '-')
            
            # companies.append({
            #     'Rank': rank,
            #     'Ticker': ticker,
            #     'Company Name': company_name,
            #     'Weight': weight
            # })
            companies.append(company_name)

    return companies

if __name__ == "__main__":
    # !pip show yfinance
    # !pip install yfinance==0.2.50
    # !pip install yfinance --upgrade

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
    
    stock_price = get_stock_data(tickers, start_date, end_date)['Close']
    stock_price = stock_price.dropna(axis=1)
    stock_returns = stock_price.pct_change().dropna()
    print(stock_returns.head())

    market_caps = get_market_caps(tickers)
    print("Sorted stocks by market cap:", market_caps)

    sp500_companies = fetch_sp500_companies()
    print(len(sp500_companies), sp500_companies)
    
