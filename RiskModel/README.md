# Risk Model



## Shrinkage Estimator
**shrinkage estimator** of the covariance matrix of stock returns can applied to improve estimation stability and portfolio optimization. This approach addresses challenges such as noise in high-dimensional financial data and the non-invertibility of covariance matrices when the number of assets exceeds observations.

The shrinkage estimator blends the **sample covariance matrix** with a **scaled identity matrix** or **average correlation structure** as the shrinkage target:

$$
\begin{align*}
S_{\text{shrunk}} = (1 - \beta) S_{\text{target}} + \beta S_{\text{sample}}
\end{align*}
$$

where:
- $S_{\text{shrunk}}$ is the shrinkage covariance matrix.
- $S_{\text{target}}$ is the shrinkage target (either `identity` or `avgcorr`).
- $S_{\text{sample}}$ is the sample covariance matrix.
- $\beta$ is the shrinkage intensity ($\beta = 1$ uses the sample covariance (no shrinkage); $\beta = 0$ applies full shrinkage).

### **Shrinkage Targets**
1. **Identity Matrix (`identity`)**
   $$
   \begin{align*}
   S_{\text{target}}[i, j] =
   \begin{cases}
   \sigma_i^2 & \text{if } i = j \\
   0 & \text{if } i \neq j
   \end{cases}
   \end{align*}
   $$
   where $\sigma_i^2$ is the variance of stock $i$.

2. **Average Correlation Matrix (`avgcorr`)**
   $$
   \begin{align*}
   S_{\text{target}}[i, j] =
   \begin{cases}
   \sigma_i^2 & \text{if } i = j \\
   \rho_{\text{avg}} \cdot \sigma_i \cdot \sigma_j & \text{if } i \neq j
   \end{cases}
   \end{align*}
   $$
   where:
   - $\rho_{\text{avg}}$ is the average pairwise correlation.
   - $\sigma_i, \sigma_j$ are standard deviations of stocks $i$ and $j$.

---

### **Shrinkage Slope ($\beta$)**
The shrinkage intensity $\beta$ determines the weight between the sample covariance and the shrinkage target:

$$
\begin{align*}
\beta = \frac{\delta^2}{\omega^2 + \delta^2} = 1 - \frac{\omega^2}{\omega^2 + \delta^2}
\end{align*}
$$

where:
- $\omega^2$ is the **estimation error** on the sample covariance matrix.
- $\delta^2$ is the **dispersion of the covariance elements**.

---

### **Estimating $\omega^2$ (Estimation Error)**
The estimation error $\omega^2$ measures how far the sample covariance matrix $S_{\text{sample}}$ deviates from the true (but unknown) covariance matrix $\Sigma$. 
$$
\begin{align*}
\omega^2 = E[||S_{\text{sample}} - \Sigma||^2] = \frac{1}{T(T-1)} \sum_{t=1}^{T} || X_t X'_t - S_{\text{sample}} ||^2
\end{align*}
$$
where:
- $T$ is the number of observations.
- $X_t X'_t$ is the outer product of the return vector at time $t$.

---

### **Estimating $\delta^2$ (Dispersion)**
The dispersion $\delta^2$ measures how far the true covariance matrix $\Sigma$ deviates from the shrinkage target $S_{\text{target}}$. Using the decomposition:

$$
\begin{align*}
E[\| S_{\text{sample}} - S_{\text{target}} \|^2] = E[\| S_{\text{sample}} - \Sigma \|^2] + \| \Sigma - S_{\text{target}} \|^2 = \omega^2 + \delta^2
\end{align*}
$$

Since $E[\| S_{\text{sample}} - S_{\text{target}} \|^2] \approx \| S_{\text{sample}} - S_{\text{target}} \|^2$,
 
$$
\begin{align*}
\hat{\delta}^2 = \| S_{\text{sample}} - S_{\text{target}} \|^2 - \hat{\omega}^2
\end{align*}
$$

where:
- $S_{\text{sample}}$ is the sample covariance matrix.
- $S_{\text{target}}$ is the shrinkage target (`identity` or `avgcorr`).
- $\hat{\omega}^2$ is the estimated covariance estimation error.

If the shrinkage target is the **scaled identity matrix**, $S_{\text{target}} = \bar{\sigma} I$, where $\bar{\sigma} = \frac{\text{trace}(S_{\text{sample}})}{n}$:
$$
\hat{\delta}^2 = \| S_{\text{sample}} - \bar{\sigma} I \|^2 - \hat{\omega}^2
$$

If the shrinkage target uses the **average correlation structure**, adjust $S_{\text{target}}$ accordingly based on the correlations.


---

## Estimation Advantages
- **Handles Non-Invertible Covariance Matrices:** Ensures the covariance matrix remains invertible for portfolio optimization.
- **Reduces Systematic Bias:** Mitigates overweighting stocks with artificially low risk estimates.
- **Improves Stability:** Balances estimation accuracy and robustness by pushing extreme variances toward a neutral level.

---

## Empirical Study

### Test Environment
The test evaluates the impact of different covariance matrix estimation techniques in portfolio optimization using a rolling-window backtesting framework. The risk models compared include:
- **Sample Covariance Matrix**
- **Ledoit-Wolf Shrinkage Estimator with Average Correlation Matrix**
- **Ledoit-Wolf Shrinkage Estimator with Identity Matrix**
- **Scikit-Learn Ledoit-Wolf Shrinkage Estimator**

The backtest covers the test period from **December 2014 to December 2024**, applied to different subsets of **S&P 500 stocks** ($N = 30, 50, 100, 500$) and varying estimation window lengths ($T=12, 24, 36, 60$ months).

**SPY** (S&P 500 ETF) is used as the benchmark, and the **Information Ratio (IR)** is computed relative to SPY to measure performance improvement over the market index.

### Performance Metrics
The evaluation focuses on the following key performance indicators:
- Cumulative Return (CR)
- Average Return (AR)
- Standard Deviation (STD)
- Sharpe Ratio (SR)
- Information Ratio (IR)

### Results
The following table summarizes the Standard Deviation, Sharpe Ratio and Information Ratio for selected configurations:
| Stock Count | Window Period (Months) | Risk Model            | Shrink Target Method | Standard Deviation | Sharpe Ratio | Information Ratio |
|------------|----------------------|----------------------|----------------------|--------------------|--------------|------------------|
| 30         | 12                   | Sample               |                      | 0.22               | 1.40         | 0.99             |
| 30         | 12                   | LedoitWolf           | avgcorr              | 0.22               | 1.40         | 0.99             |
| 30         | 12                   | LedoitWolf           | identity             | 0.22               | 1.43         | 1.03             |
| 30         | 12                   | LedoitWolfSkLearn    |                      | 0.22               | 1.43         | 1.03             |
| 30         | 24                   | Sample               |                      | 0.20               | 1.43         | 0.82             |
| 30         | 24                   | LedoitWolf           | avgcorr              | 0.20               | 1.43         | 0.82             |
| 30         | 24                   | LedoitWolf           | identity             | 0.20               | 1.46         | 0.86             |
| 30         | 24                   | LedoitWolfSkLearn    |                      | 0.20               | 1.46         | 0.86             |
| 30         | 36                   | Sample               |                      | 0.18               | 1.19         | 0.44             |
| 30         | 36                   | LedoitWolf           | avgcorr              | 0.18               | 1.19         | 0.44             |
| 30         | 36                   | LedoitWolf           | identity             | 0.18               | 1.21         | 0.47             |
| 30         | 36                   | LedoitWolfSkLearn    |                      | 0.18               | 1.21         | 0.47             |
| 30         | 60                   | Sample               |                      | 0.17               | 1.58         | 0.91             |
| 30         | 60                   | LedoitWolf           | avgcorr              | 0.17               | 1.58         | 0.91             |
| 30         | 60                   | LedoitWolf           | identity             | 0.17               | 1.59         | 0.93             |
| 30         | 60                   | LedoitWolfSkLearn    |                      | 0.17               | 1.59         | 0.93             |
| 50         | 12                   | Sample               |                      | 0.21               | 1.18         | 0.67             |
| 50         | 12                   | LedoitWolf           | avgcorr              | 0.21               | 1.18         | 0.67             |
| 50         | 12                   | LedoitWolf           | identity             | 0.21               | 1.22         | 0.73             |
| 50         | 12                   | LedoitWolfSkLearn    |                      | 0.21               | 1.22         | 0.73             |
| 50         | 24                   | Sample               |                      | 0.19               | 1.36         | 0.70             |
| 50         | 24                   | LedoitWolf           | avgcorr              | 0.19               | 1.36         | 0.70             |
| 50         | 24                   | LedoitWolf           | identity             | 0.19               | 1.38         | 0.74             |
| 50         | 24                   | LedoitWolfSkLearn    |                      | 0.19               | 1.38         | 0.74             |
| 50         | 36                   | Sample               |                      | 0.17               | 1.15         | 0.35             |
| 50         | 36                   | LedoitWolf           | avgcorr              | 0.17               | 1.15         | 0.35             |
| 50         | 36                   | LedoitWolf           | identity             | 0.17               | 1.17         | 0.39             |
| 50         | 36                   | LedoitWolfSkLearn    |                      | 0.17               | 1.17         | 0.39             |
| 50         | 60                   | Sample               |                      | 0.17               | 1.36         | 0.67             |
| 50         | 60                   | LedoitWolf           | avgcorr              | 0.17               | 1.36         | 0.67             |
| 50         | 60                   | LedoitWolf           | identity             | 0.17               | 1.37         | 0.68             |
| 50         | 60                   | LedoitWolfSkLearn    |                      | 0.17               | 1.37         | 0.68             |
| 100        | 12                   | Sample               |                      | 0.20               | 1.26         | 0.68             |
| 100        | 12                   | LedoitWolf           | avgcorr              | 0.20               | 1.27         | 0.69             |
| 100        | 12                   | LedoitWolf           | identity             | 0.20               | 1.26         | 0.69             |
| 100        | 12                   | LedoitWolfSkLearn    |                      | 0.20               | 1.26         | 0.69             |
| 100        | 24                   | Sample               |                      | 0.17               | 1.23         | 0.42             |
| 100        | 24                   | LedoitWolf           | avgcorr              | 0.17               | 1.23         | 0.42             |
| 100        | 24                   | LedoitWolf           | identity             | 0.17               | 1.24         | 0.44             |
| 100        | 24                   | LedoitWolfSkLearn    |                      | 0.17               | 1.24         | 0.44             |
| 100        | 36                   | Sample               |                      | 0.15               | 1.08         | 0.10             |
| 100        | 36                   | LedoitWolf           | avgcorr              | 0.15               | 1.08         | 0.10             |
| 100        | 36                   | LedoitWolf           | identity             | 0.16               | 1.11         | 0.15             |
| 100        | 36                   | LedoitWolfSkLearn    |                      | 0.16               | 1.11         | 0.15             |
| 100        | 60                   | Sample               |                      | 0.16               | 1.26         | 0.43             |
| 100        | 60                   | LedoitWolf           | avgcorr              | 0.16               | 1.26         | 0.43             |
| 100        | 60                   | LedoitWolf           | identity             | 0.16               | 1.26         | 0.44             |
| 100        | 60                   | LedoitWolfSkLearn    |                      | 0.16               | 1.26         | 0.44             |
| 503        | 12                   | Sample               |                      | 0.17               | 1.03         | 0.24             |
| 503        | 12                   | LedoitWolf           | avgcorr              | 0.17               | 1.03         | 0.24             |
| 503        | 12                   | LedoitWolf           | identity             | 0.17               | 1.06         | 0.29             |
| 503        | 12                   | LedoitWolfSkLearn    |                      | 0.17               | 1.06         | 0.29             |
| 503        | 24                   | Sample               |                      | 0.17               | 1.08         | 0.26             |
| 503        | 24                   | LedoitWolf           | avgcorr              | 0.17               | 1.07         | 0.25             |
| 503        | 24                   | LedoitWolf           | identity             | 0.17               | 1.10         | 0.29             |
| 503        | 24                   | LedoitWolfSkLearn    |                      | 0.17               | 1.10         | 0.29             |
| 503        | 36                   | Sample               |                      | 0.16               | 0.65         | -0.40            |
| 503        | 36                   | LedoitWolf           | avgcorr              | 0.15               | 0.65         | -0.40            |
| 503        | 36                   | LedoitWolf           | identity             | 0.16               | 0.67         | -0.38            |
| 503        | 36                   | LedoitWolfSkLearn    |                      | 0.16               | 0.67         | -0.37            |
| 503        | 60                   | Sample               |                      | 0.15               | 0.85         | -0.15            |
| 503        | 60                   | LedoitWolf           | avgcorr              | 0.15               | 0.85         | -0.16            |
| 503        | 60                   | LedoitWolf           | identity             | 0.15               | 0.87         | -0.13            |
| 503        | 60                   | LedoitWolfSkLearn    |                      | 0.15               | 0.86         | -0.14            |


### Observations
- **Shrinkage Improves Performance**: The Ledoit-Wolf shrinkage estimators generally outperform the sample covariance matrix in terms of Sharpe and Information Ratios.
- **Identity vs. AvgCorr Target**: The identity matrix shrinkage target slightly outperforms the average correlation matrix target in most cases.
- **Scikit-Learn Ledoit-Wolf vs. Custom Implementation**: The performance of Scikit-Learn's Ledoit-Wolf implementation is nearly identical to the custom implementation.
- **Impact of Window Size**: Larger estimation windows ($T=36, 60$) lead to more stable performance but may not always yield higher Sharpe Ratios.
- **Impact of Portfolio Size**: The portfolios with fewer stocks (e.g., 30 or 50) tend to have higher Sharpe and Information Ratios compared to larger portfolios. This is because smaller portfolios can better exploit specific stock mispricings and deliver higher risk-adjusted returns. According to Grinold and Kahn (2000, Chapter 15), in case N is very large, a manager is probably ill-advised to actively invest in all the stocks making up the index. The realized information ratio can be improved by, for example, focusing on the 50 or 100 largest stocks in the index and setting the weights of the remaining ones equal to zero.


