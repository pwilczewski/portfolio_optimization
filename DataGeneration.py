import scipy.optimize as sco
import numpy as np

# minimizing negative sharpe ratio
def objective_function(weights, mu, sigma):
    return -weights @ mu / np.sqrt(weights.T @ sigma @ weights)

def portfolio_optimization(mu, sigma):
    initial_weights = np.ones(5) / 5
    bounds = [(0, 1) for _ in range(5)]
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]
    result = sco.minimize(objective_function, initial_weights, args=(mu, sigma,), bounds=bounds, constraints=constraints)
    return result.x

def generate_data(n):

    data_x = []
    data_y = []
    sharpe_ratios = []

    for _ in range(n):
        # simulate mean returns
        mu = np.random.rand(5)/10 

        # simulate covariance matrix
        sigma = np.cov(np.random.randn(5, 5), rowvar=False)
        sigma = (sigma + sigma.T) / 2
        sigma = sigma + 1e-6*np.eye(5)
        cholesky_factor = np.linalg.cholesky(sigma)
        sigma = cholesky_factor @ cholesky_factor.T / 10
        weights = portfolio_optimization(mu, sigma)
        sharpe_ratios.append(-objective_function(weights, mu, sigma))

        #print("Optimal portfolio weights:", weights)
        #print("Portfolio return:", mu @ weights)
        #print("Sharpe ratio:", mu @ weights / np.sqrt(weights.T @ sigma @ weights))

        # process data for modeling
        data_x.append(list(mu) + list(sigma.flatten()))
        data_y.append(weights)

    return (data_x, data_y)