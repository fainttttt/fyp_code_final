import numpy as np
import pandas as pd
import os
import random
import time
from scipy.optimize import minimize

os.chdir('/Users/fabian/Desktop/FYP/FYP_Code/code')
print("Current Working Directory:", os.getcwd())

current_dir = os.getcwd()

relative_file_path = os.path.join(current_dir, 'data_clean', '9_Sectors_Ticker_Monthly_Returns.csv')
monthly_return = pd.read_csv(relative_file_path)

relative_file_path = os.path.join(current_dir, 'data_clean', 'economic_indicators.csv')
economic_indicators = pd.read_csv(relative_file_path)

relative_file_path = os.path.join(current_dir, 'data_clean', '01_HMM_Regimes_BullBear.csv')
hmm_regimes = pd.read_csv(relative_file_path)

risk_averse = 1
risk_seeking = -1 # may not be possible in mpt algorithm

# Using np.where to create a new array based on the conditions
hmm_regime_parameter = np.where(hmm_regimes.iloc[:, 0] == 'bull', risk_averse, risk_seeking)
hmm_regimes.iloc[:,0] = hmm_regime_parameter
hmm_regimes.head()

# Function to simulate returns based on the specified method
def simulate_return(historical_data, num_simulations, method='multivariate_normal'):
    np.random.seed(123)
    column_names = historical_data.columns.tolist()
    
    if method == 'multivariate_normal':
        mean_estimate = historical_data.mean().values
        cov_estimate = historical_data.cov().values
        simulated_returns = pd.DataFrame(
            np.random.multivariate_normal(mean_estimate, cov_estimate, num_simulations),
            columns = column_names
            )
        
    return simulated_returns

def create_bootstrap_samples(historical_data, num_simulations, num_samples, sample_size, method = 'multivariate_normal'):
    
    simulated_return = simulate_return(historical_data, num_simulations, method)

    bootstrap_samples = []
    for i in range(num_samples):
        np.random.seed(i)
        bootstrap_df = simulated_return.sample(sample_size,replace=True)
        bootstrap_samples.append(bootstrap_df)
        
    return(bootstrap_samples)

def calculate_bottom_mean(bootstrap_sample, weights, cutoff=0.05):
    n_samples = bootstrap_sample.shape[0]
    portfolio_returns = bootstrap_sample.values @ weights

    # Get bottom returns based on cutoff
    threshold_index = int(n_samples * cutoff)
    bottom_returns = np.sort(portfolio_returns)[:threshold_index]

    # Calculate the mean of the bottom returns
    return np.mean(bottom_returns)

def optimize_portfolio_bottom_mean(all_bootstrap_samples, cutoff=0.05):
    num_assets = all_bootstrap_samples[0].shape[1]
    
    # Initial guess for weights (equal distribution)
    initial_weights = np.ones(num_assets) / num_assets

    # Objective function to maximize the sum of bottom means
    def objective(weights):
        return -sum(calculate_bottom_mean(sample, weights, cutoff) for sample in all_bootstrap_samples)

    # Constraints: weights sum to 1 and are non-negative
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 0.3)] * num_assets  # No shorting constraint

    # Optimize using SciPy
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)

    if result.success:
        optimal_weights = result.x
        total_bottom_mean = -result.fun  # Change back to positive since we minimized
        return optimal_weights, total_bottom_mean
    else:
        print("Optimization failed:", result.message)
        return None, None

def rolling_portfolio_optimization_proposed(monthly_return_entire_data, num_simulations=500, num_samples=100, sample_size=20, simulation_method='multivariate_normal', window_size=36, alpha=0.1):
    length_data = monthly_return_entire_data.shape[0]
    all_results = []

    for start in range(length_data - window_size):
        end = start + window_size
        historical_data = monthly_return_entire_data.iloc[start:end]

        # Create bootstrap samples
        bootstrap_samples = create_bootstrap_samples(historical_data, num_simulations, num_samples, sample_size, method=simulation_method)
        
        # Optimize portfolio for bootstrap samples and store results
        all_results.append(optimize_portfolio_bottom_mean(bootstrap_samples, cutoff=alpha))
        print(f"Completed: {start}")

    return all_results

# Define Parameters
all_historical_data = monthly_return.iloc[:, :9]
num_simulations = 500
num_samples = 20
sample_size = 20
simulation_method = 'multivariate_normal'
window_size = 36
alpha = 0.2

# Usage Example
all_result = rolling_portfolio_optimization_proposed(
    monthly_return_entire_data=all_historical_data,
    num_simulations=num_simulations,
    num_samples=num_samples,
    sample_size=sample_size,
    simulation_method=simulation_method,
    window_size=window_size,
    alpha=alpha
)

# Output results
for weight, bottom_mean in all_result:
    if weight is not None:
        result = [round(num, 2) for num in weight]
        print(result)


# historical_data = monthly_return.iloc[:36, :9]  # Adjust column slicing as needed
# bootstrap_samples = create_bootstrap_samples(historical_data, num_simulations, num_samples, sample_size, method=simulation_method)
# result = optimize_portfolio_bottom_mean(bootstrap_samples, cutoff=alpha)
# print(result)

# wt = result[0]

# all_sample_bottom_mean = []
# for sample in bootstrap_samples:
#     all_sample_bottom_mean.append(np.mean(np.sort(np.array(sample @ wt))[:4]))
    
# all_sample_bottom_mean
# np.sum(all_sample_bottom_mean)
# result[1]