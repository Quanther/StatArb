import datetime as dt
import pandas as pd
from DataLoader import fetch_sp500_companies
from Optimizer_SR import backtest_portfolio
import time
import random


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
