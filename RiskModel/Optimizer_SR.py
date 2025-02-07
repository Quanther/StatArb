import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as spo
from RiskModel import RiskModel
from yfinance_loader import get_stock_data
from sklearn.covariance import LedoitWolf

def assess_portfolio(prices, allocs, cov_matrix=None):
    """
    Assess portfolio performance using given prices and allocations.

    Parameters
    ----------
    prices: DataFrame of stock prices
    allocs: List of asset allocations
    cov_matrix: Covariance matrix (sample or Ledoit-Wolf shrinkage)

    Returns
    -------
    cr: Cumulative return
    adr: Average daily return
    sddr: Standard deviation of daily returns (from covariance matrix)
    sr: Sharpe ratio
    """
    normed = prices / prices.iloc[0]
    alloced = normed * allocs
    port_val = alloced.sum(axis=1)
    daily_rets = port_val.pct_change().dropna()

    cr = port_val.iloc[-1] / port_val.iloc[0] - 1
    adr = daily_rets.mean()

    # Use sample or provided covariance matrix to calculate volatility
    if cov_matrix is None:
        cov_matrix = np.cov(daily_rets, rowvar=False)

    port_volatility = np.sqrt(np.dot(allocs.T, np.dot(cov_matrix, allocs)))
    sr = np.sqrt(252) * adr / port_volatility

    return cr, adr, port_volatility, sr

def error_fct(allocs, prices, cov_matrix):
    """
    Compute error based on the given allocations (inverse of Sharpe ratio).
    """
    _, _, _, sr = assess_portfolio(prices, allocs, cov_matrix)
    return -sr  # Inverse for minimization

def fit_alloc(prices, cov_matrix, error_fct):
    """
    Fit a portfolio allocation that minimizes the error function.
    """
    num_assets = len(prices.columns)
    ini_guess = np.array([1.0 / num_assets] * num_assets)

    # Call optimizer to minimize error function
    bnds = tuple((0, 1) for _ in range(num_assets))
    cons = ({'type': 'eq', 'fun': lambda a: 1 - np.sum(a)})
    result = spo.minimize(error_fct, 
                          ini_guess, 
                          args=(prices, cov_matrix), 
                          method='SLSQP',
                          bounds=bnds, 
                        #   options={'disp': True},
                          constraints=cons)
    return result.x

def optimize_portfolio(sd='2021-01-01', ed='2025-01-01', syms=["AAPL", "MSFT", "GOOGL", "AMZN"], risk_matrix='Sample', gen_plot=False):
    """
    Optimize the portfolio allocation to maximize the Sharpe ratio.
    """

    # Fetch stock prices
    stock_data = get_stock_data(syms, sd, ed)
    stock_data = stock_data.dropna(axis=1)
    prices = stock_data['Adj Close']
    
    returns = prices.pct_change().dropna().values

    # Apply the Ledoit-Wolf shrinkage model
    if risk_matrix == 'Sample':
        cov_matrix = np.cov(returns, rowvar=False)
    
    elif risk_matrix == 'LedoitWolf':
        rm = RiskModel()
        cov_matrix, _, _, _ = rm.shrinkage_covariance(returns=returns, shrink_target_method='identity')
    elif risk_matrix == 'LedoitWolfSkLearn':
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns).covariance_
    
    
    # Find optimal allocations
    allocs = fit_alloc(prices, cov_matrix, error_fct)

    cr, adr, sddr, sr = assess_portfolio(prices, allocs, cov_matrix)

    if gen_plot:
        normed = prices / prices.iloc[0]
        alloced = normed * allocs
        port_val = alloced.sum(axis=1)
        port_val.plot(title='Daily Portfolio Value', fontsize=12)
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        # plt.savefig('images/plot.png')
        plt.show()

    return allocs, cr, adr, sddr, sr

if __name__ == "__main__":

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)[0]
    sp500_tickers = table['Symbol'].tolist()

    url_dow = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    dow_table = pd.read_html(url_dow, header=0)[2]
    dow_tickers = dow_table['Symbol'].tolist()

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Example tickers
    tickers = sp500_tickers
    start_date = '2021-01-01'
    end_date = '2025-01-01'
    risk_matrix = "LedoitWolf" #"LedoitWolf", "LedoitWolf", "LedoitWolfSkLearn"

    allocs, cr, adr, sddr, sr = optimize_portfolio(start_date, 
                                                   end_date, 
                                                   tickers, 
                                                   risk_matrix, 
                                                   gen_plot=False)
    print(f"Optimal Allocations: {sorted([f'{t}: {a:.4f}' for t, a in zip(tickers, allocs) if round(a, 4) != 0], key=lambda x: float(x.split(': ')[1]), reverse=True)}")
    print(f"Cumulative Return: {cr:.4f}")
    print(f"Average Daily Return: {adr:.4f}")
    print(f"Std Dev: {sddr:.4f}")
    print(f"Sharpe Ratio: {sr:.4f}")
    
          
