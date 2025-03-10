import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as spo
from RiskModel import RiskModel
from DataLoader import get_stock_data, get_market_caps, fetch_sp500_companies
from sklearn.covariance import LedoitWolf
import time
import random

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

def optimize_portfolio(sd='2021-01-01', ed='2025-01-01', syms=["AAPL", "MSFT", "GOOGL", "AMZN"], risk_matrix='Sample', shrink_target_method=None, gen_plot=False):
    """
    Optimize the portfolio allocation to maximize the Sharpe ratio.
    """

    # Fetch stock prices
    stock_data = get_stock_data(syms, sd, ed)
    stock_data = stock_data.dropna(axis=1)
    prices = stock_data['Close']
    
    returns = prices.pct_change().dropna().values

    # Apply the Ledoit-Wolf shrinkage model
    if risk_matrix == 'Sample':
        cov_matrix = np.cov(returns, rowvar=False)
    
    elif risk_matrix == 'LedoitWolf':
        rm = RiskModel()
        cov_matrix, _, _, _ = rm.shrinkage_covariance(returns=returns, shrink_target_method=shrink_target_method)
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

def compute_information_ratio(portfolio_return, benchmark_return, portfolio_std):
    active_return = portfolio_return - benchmark_return
    tracking_error = portfolio_std  # Assuming tracking error ≈ portfolio std deviation
    return active_return / tracking_error if tracking_error > 0 else np.nan

def backtest_portfolio(sd='2014-12-31', ed='2024-12-31', tickers=["AAPL", "MSFT", "GOOGL", "AMZN"], 
                        risk_matrix='Sample', shrink_target_method=None, window_size_month = 12, step_size_month = 1,
                        benchmark_ticker="SPY"):
    """
    Backtest portfolio optimization using a rolling window approach.
    """
    sd_fetch = (dt.datetime.strptime(sd, "%Y-%m-%d") - pd.DateOffset(months=window_size_month)).strftime("%Y-%m-%d")
    stock_data = get_stock_data(tickers, sd_fetch, ed)
    benchmark_data = get_stock_data([benchmark_ticker], sd, ed)
    
    prices = stock_data['Close'].dropna(axis=1)
    benchmark_prices = benchmark_data['Close'][benchmark_ticker].dropna()
    dates = prices.groupby([prices.index.year, prices.index.month]).tail(1).index

    results = []
    portfolio_returns = []
    benchmark_returns_series = []

    for start_idx in range(0, len(dates) - window_size_month - step_size_month, step_size_month):
        train_start_date = dates[start_idx]
        test_start_date = dates[start_idx + window_size_month]
        train_end_date = test_start_date-pd.DateOffset(days=1)
        test_end_date = min(dates[start_idx + window_size_month + step_size_month]-pd.DateOffset(days=1), prices.index[-1])
        
        # Training and test window data
        train_prices = prices.loc[train_start_date:train_end_date]
        test_prices = prices.loc[test_start_date:test_end_date]
        benchmark_test_prices = benchmark_prices.loc[test_start_date:test_end_date]

        train_returns = train_prices.pct_change().dropna().values

        # Estimate covariance matrix
        if risk_matrix == 'Sample':
            cov_matrix = np.cov(train_returns, rowvar=False)
        elif risk_matrix == 'LedoitWolf':
            rm = RiskModel()
            cov_matrix, _, _, _ = rm.shrinkage_covariance(returns=train_returns, shrink_target_method=shrink_target_method)
        elif risk_matrix == 'LedoitWolfSkLearn':
            lw = LedoitWolf()
            cov_matrix = lw.fit(train_returns).covariance_

        # Find optimal allocations
        allocs = fit_alloc(train_prices, cov_matrix, error_fct)

        # Evaluate performance on the test set
        cr, adr, sddr, sr = assess_portfolio(test_prices, allocs, cov_matrix)

        # Compute test set returns for IC and IR
        test_returns = test_prices.pct_change().dropna().dot(allocs)
        portfolio_returns.extend(test_returns)
        benchmark_returns = benchmark_test_prices.pct_change().dropna()
        benchmark_returns_series.extend(benchmark_returns)
        benchmark_cr = (benchmark_test_prices.iloc[-1] / benchmark_test_prices.iloc[0]) - 1

        # Information Ratio (IR)
        active_returns = test_returns - benchmark_returns
        ir = active_returns.mean() * np.sqrt(252) / active_returns.std() if active_returns.std() > 0 else None

        # Information Coefficient (IC)
        ic = np.corrcoef(test_returns, benchmark_returns)[0, 1] if len(test_returns) > 0 else None

        results.append({
            "Train Start": train_start_date,
            "Train End": train_end_date,
            "Test Start": test_start_date,
            "Test End": test_end_date,
            "Cumulative Return": cr,
            "Benchmark Cumulative Return": benchmark_cr,
            "Average Daily Return": adr,
            "Standard Deviation": sddr,
            "Sharpe Ratio": sr,
            "Information Ratio": ir,
            "Information Coefficient": ic,
            "Optimal Allocations": [
                f"{ticker}: {alloc:.4f}" for ticker, alloc in zip(tickers, allocs) if round(alloc, 4) != 0
            ]
        })

        print(f'Train from {train_start_date} to {train_end_date} and Test from {test_start_date} to {test_end_date} completed...')
    
    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results["Optimal Allocations"] = df_results["Optimal Allocations"].apply(
        lambda x: ", ".join(sorted(x, key=lambda alloc: float(alloc.split(": ")[1]), reverse=True)))
    
    cum_return = df_results['Cumulative Return'].apply(lambda x: x+1).prod()-1
    annualized_return = (cum_return+1)**(12/len(df_results))-1
    annualized_std_dev = df_results['Cumulative Return'].std()*np.sqrt(12)
    sharpe_ratio = annualized_return / annualized_std_dev

    active_returns = df_results['Cumulative Return'] - df_results['Benchmark Cumulative Return']
    cum_active_return = active_returns.apply(lambda x: x+1).prod()-1
    annualized_active_return = (cum_active_return+1)**(12/len(df_results))-1
    annualized_active_std_dev = active_returns.std()*np.sqrt(12)
    ir = annualized_active_return / annualized_active_std_dev

    # Compute Information Coefficient (IC)
    breadth = 12 * len(tickers)
    # ic = np.corrcoef(portfolio_returns, benchmark_returns_series)[0, 1] if len(portfolio_returns) > 0 else None
    ic = ir / np.sqrt(breadth)

    print(f"Cumulative Return: {cum_return:.4f}")
    print(f"Annualized Return: {annualized_return:.4f}")
    print(f"Standard Deviation: {annualized_std_dev:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Information Ratio (IR): {ir:.4f}")
    print(f"Information Coefficient (IC): {ic:.4f}")

    if risk_matrix == 'LedoitWolf':
        filename = f'backtest_results_N={len(tickers)}_wd={window_size_month}_rm={risk_matrix}+{shrink_target_method}.csv'
    else:
        filename = f'backtest_results_N={len(tickers)}_wd={window_size_month}_rm={risk_matrix}.csv'
    df_results.to_csv('result/' + filename, index=False)

    print(f"Backtesting completed. Results saved to {filename}")

    return cum_return, annualized_return, annualized_std_dev, sharpe_ratio, ir, ic

if __name__ == "__main__":

    """ Rolling window back-testing """
    
    sp500_tickers = fetch_sp500_companies()
    
    # Parameters for testing
    start_date = '2014-12-31'
    end_date = '2024-12-31'
    stock_counts = [30, 50, 100, len(sp500_tickers)]
    window_periods = [12, 24, 36, 60]
    risk_models = ["Sample", "LedoitWolf", "LedoitWolfSkLearn"]
    shrink_target_methods = ["identity", "avgcorr"]  # Example methods, adjust based on your code
    results = []
    delay=2

    # stock_count=len(sp500_tickers); wd=60; risk_matrix="LedoitWolfSkLearn"; shrink_target_method="avgcorr"
    
    # Run tests and collect results
    for stock_count in stock_counts:
        tickers_subset = sp500_tickers[:stock_count]
        
        for wd in window_periods:

            for risk_matrix in risk_models:
                if risk_matrix == "LedoitWolf":
                    for shrink_target_method in shrink_target_methods:
                        print(f"Testing [{stock_count}] stocks with window [{wd}] months using [{risk_matrix}] with target method [{shrink_target_method}]...")
                        
                        try:
                            cr, ar, astd, sr, ir, ic = backtest_portfolio(start_date, end_date, tickers_subset, risk_matrix, shrink_target_method, wd)
                            
                            result = {
                                "Stock Count": stock_count,
                                "Window Period (Months)": wd,
                                "Risk Model": risk_matrix,
                                "Shrink Target Method": shrink_target_method,
                                "Cumulative Return": cr,
                                "Average Return": ar,
                                "Standard Deviation": astd,
                                "Sharpe Ratio": sr,
                                "Information Ratio": ir,
                                "Information Coefficient": ic
                            }
                            results.append(result)
                            
                            print(f"Test completed: {result}")

                        except Exception as e:
                            print(f"Error for {stock_count} stocks, {wd} months, {risk_matrix}, {shrink_target_method}: {e}")

                    time.sleep(delay + random.uniform(0, 1))

                else:
                    print(f"Testing [{stock_count}] stocks with window [{wd}] months using [{risk_matrix}]...")
                        
                    try:
                        cr, ar, astd, sr, ir, ic = backtest_portfolio(start_date, end_date, tickers_subset, risk_matrix, None, wd)
                        
                        result = {
                            "Stock Count": stock_count,
                            "Window Period (Months)": wd,
                            "Risk Model": risk_matrix,
                            "Shrink Target Method": '',
                            "Cumulative Return": cr,
                            "Average Return": ar,
                            "Standard Deviation": astd,
                            "Sharpe Ratio": sr,
                            "Information Ratio": ir,
                            "Information Coefficient": ic
                        }
                        results.append(result)
                        
                        print(f"Test completed: {result}")

                    except Exception as e:
                        print(f"Error for {stock_count} stocks, {wd} months, {risk_matrix}: {e}")

                    time.sleep(delay + random.uniform(0, 1))
                    

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv("result/backtesting_result_summary_rolling_window.csv", index=False, header=False, mode='a')

    print("All tests completed. Results saved to 'backtesting_result_summary.csv'")

