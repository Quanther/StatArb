import datetime as dt
import pandas as pd
from DataLoader import fetch_sp500_companies
from Optimizer_SR import optimize_portfolio, compute_information_ratio
import time
import random


sp500_tickers = fetch_sp500_companies()

# Parameters for testing
end_date = '2025-01-01'
stock_counts = [30, 50, 100, len(sp500_tickers)]
years = [1, 3, 5, 10]
risk_models = ["Sample", "LedoitWolf", "LedoitWolfSkLearn"]
shrink_target_methods = ["identity", "avgcorr"]  # Example methods, adjust based on your code
results = []
delay=2

# stock_count=len(sp500_tickers); period=10; risk_matrix="Sample"; shrink_target_method="identity"

# Run tests and collect results
for stock_count in stock_counts:
    tickers_subset = sp500_tickers[:stock_count]
    
    for period in years:
        start_date = (dt.datetime.strptime(end_date, "%Y-%m-%d") - pd.DateOffset(years=period)).strftime("%Y-%m-%d")
        spy_allocs, cr_spy, adr_spy, sddr_spy, sr_spy = optimize_portfolio(start_date, end_date, ['SPY'])

        for risk_matrix in risk_models:
            if risk_matrix == "LedoitWolf":
                for shrink_target_method in shrink_target_methods:
                    print(f"Testing [{stock_count}] stocks for [{period}] years using [{risk_matrix}] with target method [{shrink_target_method}]...")
                    
                    try:
                        allocs, cr, adr, sddr, sr = optimize_portfolio(
                            start_date, end_date, tickers_subset, risk_matrix, shrink_target_method=shrink_target_method, gen_plot=False
                        )

                        ir = compute_information_ratio(adr, adr_spy, sddr)
                        
                        result = {
                            "Stock Count": stock_count,
                            "Period (Years)": period,
                            "Risk Model": risk_matrix,
                            "Shrink Target Method": shrink_target_method,
                            "Cumulative Return": cr,
                            "Average Daily Return": adr,
                            "Standard Deviation": sddr,
                            "Sharpe Ratio": sr,
                            "Information Ratio": ir,
                            "Optimal Allocations": [
                                f"{ticker}: {alloc:.4f}"
                                for ticker, alloc in zip(tickers_subset, allocs)
                                if round(alloc, 4) != 0
                            ]
                        }
                        results.append(result)
                        
                        print(f"Test completed: {result}")

                    except Exception as e:
                        print(f"Error for {stock_count} stocks, {period} years, {risk_matrix}, {shrink_target_method}: {e}")

                time.sleep(delay + random.uniform(0, 1))

            else:
                print(f"Testing [{stock_count}] stocks for [{period}] years using [{risk_matrix}]...")
                    
                try:
                    allocs, cr, adr, sddr, sr = optimize_portfolio(
                        start_date, end_date, tickers_subset, risk_matrix, shrink_target_method=shrink_target_method, gen_plot=False
                    )
                    
                    ir = compute_information_ratio(adr, adr_spy, sddr)
                    
                    result = {
                        "Stock Count": stock_count,
                        "Period (Years)": period,
                        "Risk Model": risk_matrix,
                        "Shrink Target Method": '',
                        "Cumulative Return": cr,
                        "Average Daily Return": adr,
                        "Standard Deviation": sddr,
                        "Sharpe Ratio": sr,
                        "Information Ratio": ir,
                        "Optimal Allocations": [
                            f"{ticker}: {alloc:.4f}"
                            for ticker, alloc in zip(tickers_subset, allocs)
                            if round(alloc, 4) != 0
                        ]
                    }
                    results.append(result)
                    
                    print(f"Test completed: {result}\n")

                except Exception as e:
                    print(f"Error for {stock_count} stocks, {period} years, {risk_matrix}, {shrink_target_method}: {e}")
                
                time.sleep(delay + random.uniform(0, 1))
                

# Save results to CSV
df_results = pd.DataFrame(results)
df_results["Optimal Allocations"] = df_results["Optimal Allocations"].apply(
    lambda x: ", ".join(sorted(x, key=lambda alloc: float(alloc.split(": ")[1]), reverse=True)))
df_results.to_csv("result/portfolio_optimization_results_in-sample.csv", index=False)

print("All tests completed. Results saved to 'portfolio_optimization_results.csv'")
