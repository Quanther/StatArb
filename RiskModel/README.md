# Risk Model Summary

The risk model in this project incorporates a **shrinkage estimator** for the covariance matrix of stock returns to improve estimation stability and portfolio optimization. This approach addresses challenges such as noise in high-dimensional financial data and the non-invertibility of covariance matrices when the number of assets exceeds observations.

## Shrinkage Estimator
The shrinkage estimator blends the **sample covariance matrix** with a **scaled identity matrix** or **average correlation structure** as the shrinkage target:

- **Sample Covariance:** Direct estimation based on return data.
- **Shrinkage Target (`Identity`)**: Scaled identity matrix with variances on the diagonal and zeros elsewhere.
- **Shrinkage Target (`AvgCorr`)**: Incorporates the average correlation between stocks, scaled by their standard deviations.

The final shrinkage estimator formula is:
\[
S_{\text{shrunk}} = (1 - \beta) \times S_{\text{target}} + \beta \times S_{\text{sample}}
\]
where:
- \(\beta\) is the shrinkage slope.
- \(\beta = 1\) uses the sample covariance; \(\beta = 0\) applies full shrinkage.

## Estimation Advantages
- **Handles Non-Invertible Covariance Matrices:** Ensures the covariance matrix remains invertible for portfolio optimization.
- **Reduces Systematic Bias:** Mitigates overweighting stocks with artificially low risk estimates.
- **Improves Stability:** Balances estimation accuracy and robustness by pushing extreme variances toward a neutral level.

## Performance Testing Plan
Results will be tested on portfolios of **30, 50, 100, and 500 stocks** over different horizons (1, 3, 5, and 10 years) for:
- **Sample Covariance Matrix**
- **Ledoit-Wolf with AvgCorr Shrinkage Target**
- **Ledoit-Wolf with Identity Shrinkage Target**

Stay tuned for detailed test results and performance comparisons.
