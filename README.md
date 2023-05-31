# Approximately optimal portfolios

This repo contains my research into applying DNNs to Modern Portfolio Theory. Specifically I'm investigating whether DNNs can provide a suitable differentiable approximation for optimal portfolios. In my code I consider a 5 asset portfolio. In _DataGeneration.py_ I simulate expected returns, a covariance matrix and calculate the portfolio weights that provide the optimal Sharpe Ratio. In _EstimateModel.py_ I estimate a DNN that takes the expected returns and covariance matrix as inputs and outputs the approximate portfolio weights. In _PostEstimation.ipynb_ I evaluate the model performance in-sample.

My analysis of this research is available in my [substack](https://indiequant.substack.com/p/approximately-optimal-portfolios).
