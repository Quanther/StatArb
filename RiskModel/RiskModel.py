import numpy as np

class RiskModel:
    def __init__(self):
        pass

    def shrinkage_covariance(self, returns, shrink_target_method='avgcorr', market_returns=None, cap=None):
        """
        Calculate Shrinkage Estimator of the Covariance Matrix

        :param:
            returns: returns of assets (T,n)
            shrink_target_method: matrix used in determining shrinkage target
                'avgcorr':  variance (diagonal) + average correlation * std_i * std_j (off-diagonal)
                'identity': variance (diagonal) + 0 (off-diagonal)
            market_returns: Optional, market returns (T,)
            cap: Optional, market capitalization (n,) for weighted market return if market_returns is not provided
        :return:
            S_hat: shrinkage estimator of the covariance matrix
            corr_avg: sample average correlation
            beta_hat: shrinkage slope
            betas: market betas for each asset
        """

        T, n = returns.shape
        return_mean = np.mean(returns, axis=0, keepdims=True)
        returns -= return_mean
        cov_var_sample = np.matmul(returns.T, returns) / T
        corr_avg = 0

        if shrink_target_method == 'identity':
            var_mean = np.diag(cov_var_sample).mean()
            shrink_target = var_mean * np.eye(n)

            omega_hat_squared_sum = 0.0
            for t in range(T):
                X_t = returns[t, :].reshape(-1, 1)
                S_t = np.matmul(X_t, X_t.T)
                omega_hat_squared_sum += np.linalg.norm(S_t - cov_var_sample, 'fro') ** 2
            omega_hat_squared = omega_hat_squared_sum / (T * (T - 1))

            beta_hat = 1 - omega_hat_squared / np.linalg.norm(cov_var_sample - shrink_target, 'fro') ** 2

        else:
            std_sample = np.sqrt(np.diag(cov_var_sample).reshape(-1, 1))
            cov_var_unit = np.matmul(std_sample, std_sample.T)
            corr_avg = ((cov_var_sample / cov_var_unit).sum() - n) / (n * (n - 1))
            shrink_target = corr_avg * cov_var_unit
            np.fill_diagonal(shrink_target, std_sample ** 2)

            y = returns ** 2
            phi_mat = np.matmul(y.T, y) / T - cov_var_sample ** 2
            phi = phi_mat.sum()

            theta_mat = np.matmul((returns ** 3).T, returns) / T - std_sample ** 2 * cov_var_sample
            np.fill_diagonal(theta_mat, 0)
            rho = np.diag(phi_mat).sum() + corr_avg * (1 / np.matmul(std_sample, std_sample.T) * theta_mat).sum()

            gamma = np.linalg.norm(cov_var_sample - shrink_target, "fro") ** 2
            kappa = (phi - rho) / gamma
            beta_hat = 1 - max(0, min(1, kappa / T))

        S_hat = (1 - beta_hat) * shrink_target + beta_hat * cov_var_sample

        # Calculate market returns if not provided, using cap-weighted approach
        if market_returns is None and cap is not None:
            # Normalize market caps to get weights for each time step (T, n)
            weights = cap / np.sum(cap, axis=1, keepdims=True)  
            
            # Compute market returns as a weighted average for each time step (T,)
            market_returns = np.sum(returns * weights, axis=1)  

        # Calculate betas (if market_returns is available)
        if market_returns is not None:
            market_var = np.var(market_returns)
            cov_with_market = np.cov(returns.T, market_returns, ddof=0)[0:n, -1]
            betas = (cov_with_market / market_var).reshape(-1,1)
        else:
            betas = None  # If no market returns or cap are provided, return None

        return S_hat, corr_avg, beta_hat, betas

