  		  	   		  		 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt  		  	   		  		 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		  		 		  		  		    	 		 		   		 		  
from util import get_data, plot_data
import scipy.optimize as spo
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def assess_portfolio(prices, allocs):
    """
   Parameters
    ----------
    allocs: tuple/list/array of the allocation of each asset
    prices: 2D DataFrame where each row is prices of portfolio assets

    :return: A tuple containing the cumulative return, average daily returns,
        standard deviation of daily returns and Sharpe ratio
    :rtype: tuple
    """
    normed = prices / prices.iloc[0]
    alloced = normed * allocs
    pos_vals = alloced * 1
    port_val = pos_vals.sum(axis=1)
    daily_rets = port_val / port_val.shift(1) - 1

    cr, adr, sddr, sr = [
        port_val.iloc[-1] / port_val.iloc[0] - 1,
        daily_rets.mean(),
        daily_rets.std(),
        np.sqrt(252) * daily_rets.mean() / daily_rets.std(),
    ]

    return cr, adr, sddr, sr

def error_fct(allocs, prices):  # error function
    """Compute error based on the given allocations

    Parameters
    ----------
    allocs: tuple/list/array of the allocation of each asset
    prices: 2D DataFrame where each row is prices of portfolio assets

    Returns error as a single real value.
    """

    cr, adr, sddr, sr = assess_portfolio(prices, allocs)
    # Metric: inverse of Sharpe ratio
    err = sr * -1
    return err


def fit_alloc(prices, error_fct):
    """Fit a line to given data, using a supplied error function.

    Parameters
    ----------
    prices: 2D array where each row is a point (X0, Y)
    error_fct: function that computes the inverse of Sharpe ratio as an error

    Returns allocations that minimizes the error function.
    """
    # Generate initial guess for line model
    iniGuess = np.asarray([1/len(prices.iloc[0])*1.0 for _ in range(len(prices.iloc[0]))])

    # Call optimizer to minimize error function
    bnds = tuple((0, 1) for _ in range(len(prices.iloc[0])))
    cons = ({'type': 'eq', 'fun': lambda a: 1 - np.sum(a)})
    result = spo.minimize(error_fct, iniGuess, args=(prices,), method='SLSQP', options={'disp': True},
                          bounds=bnds, constraints=cons)
    return result.x

def optimize_portfolio(
    sd=dt.datetime(2008, 1, 1),  		  	   		  		 		  		  		    	 		 		   		 		  
    ed=dt.datetime(2009, 1, 1),  		  	   		  		 		  		  		    	 		 		   		 		  
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		  		 		  		  		    	 		 		   		 		  
    gen_plot=False,  		  	   		  		 		  		  		    	 		 		   		 		  
):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Find the optimal allocations for a given set of stocks.
    The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		  		 		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. 		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio
    :type syms: list  		  	   		  		 		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. 	    	 		 		   		 		  
    :type gen_plot: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		  		 		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Read in adjusted closing prices for given symbols, date range  		  	   		  		 		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY

    prices_all.fillna(method="ffill", inplace=True)
    prices_all.fillna(method="bfill", inplace=True)

    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all["SPY"]  # only SPY, for reference
  		  	   		  		 		  		  		    	 		 		   		 		  
    # find the allocations for the optimal portfolio
    allocs = fit_alloc(prices, error_fct)

    # Get daily portfolio value
    normed = prices / prices.iloc[0]
    alloced = normed * allocs
    pos_vals = alloced * 1
    port_val = pos_vals.sum(axis=1)

    cr, adr, sddr, sr = assess_portfolio(prices, allocs)
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Compare daily portfolio value with SPY using a normalized plot  		  	   		  		 		  		  		    	 		 		   		 		  
    if gen_plot:  		  	   		  		 		  		  		    	 		 		   		 		  

        df_temp = pd.concat(  		  	   		  		 		  		  		    	 		 		   		 		  
            [port_val, prices_SPY/prices_SPY.iloc[0]], keys=["Portfolio", "SPY"], axis=1
        )  		  	   		  		 		  		  		    	 		 		   		 		  
        # plot_data(df_temp, title='Daily Portfolio Value and SPY')
        ax = df_temp.plot(title='Daily Portfolio Value and SPY', fontsize = 12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        plt.savefig('images/plot.png')
  		  	   		  		 		  		  		    	 		 		   		 		  
    return allocs, cr, adr, sddr, sr
  		  	   		  		 		  		  		    	 		 		   		 		  
def test_code():  		  	   		  		 		  		  		    	 		 		   		 		  
     		  	   		  		 		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ["IBM", "X", "GLD", "JPM"]
  		  	   		  		 		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		  		 		  		  		    	 		 		   		 		  
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Start Date: {start_date}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"End Date: {end_date}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Symbols: {symbols}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return: {adr}")  		  	   		  		 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return: {cr}")  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    test_code()  		  	   		  		 		  		  		    	 		 		   		 		  
