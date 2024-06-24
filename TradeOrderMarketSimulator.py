  		  	   		  		 		  		  		    	 		 		   		 		  
import datetime as dt
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		  		 		  		  		    	 		 		   		 		  
from util import get_data, plot_data  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def compute_portvals(  		  	   		  		 		  		  		    	 		 		   		 		  
    orders_file="./orders/orders.csv",  		  	   		  		 		  		  		    	 		 		   		 		  
    start_val=1000000,  		  	   		  		 		  		  		    	 		 		   		 		  
    commission=9.95,  		  	   		  		 		  		  		    	 		 		   		 		  
    impact=0.005,  		  	   		  		 		  		  		    	 		 		   		 		  
):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param orders_file: Path of the order file or the file object  		  	   		  		 		  		  		    	 		 		   		 		  
    :type orders_file: str or file object  		  	   		  		 		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		  		 		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		  		 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		  		 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		  		 		  		  		    	 		 		   		 		  
    """
   
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values='nan')
    orders_df = orders_df.sort_index()

    start_date = orders_df.index[0]
    end_date = orders_df.index[-1]
    symbols = orders_df.Symbol.unique()

    prices = get_data(symbols, pd.date_range(start_date, end_date))
    prices['Cash'] = 1.0

    trades = prices.copy()
    trades.loc[:,:] = 0

    for i in range(len(orders_df)):
        idx, order = orders_df.index[i], orders_df.iloc[i]
        if order.Order == 'BUY': sign = 1
        else: sign = -1
        trades.loc[idx, order.Symbol] += sign * order.Shares
        #Cash calculaiton includes the negative market impact and commission fees
        trades.loc[idx, 'Cash'] += (-sign - impact) * order.Shares * prices.loc[idx, order.Symbol] - commission 
    
    #Holding position calculated by cumulative sum of trade orders and the initial postion is added with 0 shares on stocks and the starting value of the cash
    holdings = trades.cumsum(axis=0) + np.array((trades.shape[1]-1)*[0] + [start_val])
    #Values of each stock and cash is calculated by multiplying the price and the position at each time of trade order
    values = prices * holdings
    portvals = pd.DataFrame(values.sum(axis=1), columns=['PortVal'])

    return portvals
  		  	   		  		 		  		  		    	 		 		   		 		  
def test_code():

    of = "./orders/orders.csv"
    sv = 1000000  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Process orders  		  	   		  		 		  		  		    	 		 		   		 		  
    portvals = compute_portvals(orders_file=of, start_val=sv)  		  	   		  		 		  		  		    	 		 		   		 		  
    if isinstance(portvals, pd.DataFrame):  		  	   		  		 		  		  		    	 		 		   		 		  
        portvals = portvals[portvals.columns[0]]  # just get the first column  		  	   		  		 		  		  		    	 		 		   		 		  
    else:  		  	   		  		 		  		  		    	 		 		   		 		  
        "warning, code did not return a DataFrame"  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Get portfolio stats  		  	   		  		 		  		  		    	 		 		   		 		  
    def getPortStats(portvals, benchmark='SPY', gen_plot=False):
        start_date = portvals.index[0]
        end_date = portvals.index[-1]

        daily_rets = (portvals / np.roll(portvals, 1) - 1)[1:]
        cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio, ending_value = [
            portvals[-1]/portvals[0] - 1,
            daily_rets.mean(),
            daily_rets.std(ddof=1),
            np.sqrt(252) * daily_rets.mean() / daily_rets.std(ddof=1),
            portvals[-1],
        ]

        import analysis
        portvals_BM = analysis.get_portval(
            sd=start_date,
            ed=end_date,
            syms=[benchmark],
            allocs=[1],
            sv=sv
        )
        cum_ret_BM, avg_daily_ret_BM, std_daily_ret_BM, sharpe_ratio_BM, ending_value_BM = analysis.assess_portfolio(
            sd=start_date,
            ed=end_date,
            syms=[benchmark],
            allocs=[1],
            sv=sv,
            gen_plot=False,
        )

        # Compare portfolio against $SPX
        print(f"Date Range: {start_date} to {end_date}\n")
        print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
        print(f"Sharpe Ratio of {benchmark} : {sharpe_ratio_BM}\n")
        print(f"Cumulative Return of Fund: {cum_ret}")
        print(f"Cumulative Return of {benchmark} : {cum_ret_BM}\n")
        print(f"Standard Deviation of Fund: {std_daily_ret}")
        print(f"Standard Deviation of {benchmark} : {std_daily_ret_BM}\n")
        print(f"Average Daily Return of Fund: {avg_daily_ret}")
        print(f"Average Daily Return of {benchmark} : {avg_daily_ret_BM}\n")
        print(f"Final Portfolio Value: {ending_value}")
        print(f"Final {benchmark} Value: {ending_value_BM}\n")

        if gen_plot:
            portvals_normed = portvals / portvals[0]
            portvals_BM_normed = pd.DataFrame(portvals_BM / portvals_BM[0], columns=['Benchmark'], index=portvals.index)
            df_temp = pd.concat(
                [portvals_normed, portvals_BM_normed], axis=1
            )
            ax = df_temp.plot(title='Portfolio Value with Benchmark', fontsize = 12)
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            plt.show()

    getPortStats(portvals, benchmark='$SPX', gen_plot=True)

  		  	   		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    test_code()  		  	   		  		 		  		  		    	 		 		   		 		  
