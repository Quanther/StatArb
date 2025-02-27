# Risk Model Summary

The risk model in this project incorporates a **shrinkage estimator** for the covariance matrix of stock returns to improve estimation stability and portfolio optimization. This approach addresses challenges such as noise in high-dimensional financial data and the non-invertibility of covariance matrices when the number of assets exceeds observations.

## Shrinkage Estimator
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

## Performance Testing Plan
Results will be tested on portfolios of **30, 50, 100, and 500 stocks** over different horizons (1, 3, 5, and 10 years) for:
- **Sample Covariance Matrix**
- **Ledoit-Wolf with AvgCorr Shrinkage Target**
- **Ledoit-Wolf with Identity Shrinkage Target**

Stay tuned for detailed test results and performance comparisons.
