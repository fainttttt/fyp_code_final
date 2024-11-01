import numpy as np
import pandas as pd
import os
import random
import time
from scipy.optimize import minimize
from data_simulation import *

os.chdir('/Users/fabian/Desktop/FYP/FYP_Code/code')
print("Current Working Directory:", os.getcwd())

current_dir = os.getcwd()

relative_file_path = os.path.join(current_dir, 'data_clean', '9_Sectors_Ticker_Monthly_Returns.csv')
monthly_return = pd.read_csv(relative_file_path)

relative_file_path = os.path.join(current_dir, 'data_clean', '01_HMM_Regimes_BullBear.csv')
hmm_regimes = pd.read_csv(relative_file_path)

# Function to maximize top mean
def calculate_top_mean(bootstrap_sample, weights, cutoff=0.05):
    n_samples = bootstrap_sample.shape[0]
    portfolio_returns = bootstrap_sample.values @ weights

    # Get top returns based on cutoff
    threshold_index = int(n_samples * cutoff)
    top_returns = np.sort(portfolio_returns)[-threshold_index:]

    # Calculate the mean of the top returns
    return np.mean(top_returns)

def optimize_portfolio_top_mean(all_bootstrap_samples, cutoff=0.05):
    num_assets = all_bootstrap_samples[0].shape[1]
    
    # Initial guess for weights (equal distribution)
    initial_weights = np.ones(num_assets) / num_assets

    # Objective function to maximize the sum of top means
    def objective(weights):
        return -sum(calculate_top_mean(sample, weights, cutoff) for sample in all_bootstrap_samples)

    # Constraints: weights sum to 1 and are non-negative
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 0.3)] * num_assets  # No shorting constraint

    # Optimize using SciPy
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)

    if result.success:
        optimal_weights = result.x
        total_top_mean = -result.fun  # Change back to positive since we minimized
        return optimal_weights, total_top_mean
    else:
        print("Optimization failed:", result.message)
        return None, None

# Function to maximize bottom mean
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

# Function to execute rolling window - Toggle to maximize top or bottom mean
def rolling_portfolio_optimization_proposed(monthly_return_entire_data, 
                                            num_simulations=500, num_samples=100, sample_size=20, simulation_method='multivariate_normal', df = 5, 
                                            window_size=36, alpha=0.1, risk_aversion_method = 'most_recent_performance'):
    length_data = monthly_return_entire_data.shape[0]
    all_results = [] # weight, objective value
    risk_attitudes = [] # True if risk seeking, False if risk averse

    if risk_aversion_method == 'most_recent_performance':
        risk_seeking = False
    elif risk_aversion_method == 'hidden_markov_model':
        risk_seeking = False
    elif risk_aversion_method == 'all_risk_seeking':
        risk_seeking = True
    elif risk_aversion_method == 'all_risk_averse':
        risk_seeking = False
    else:
        print('Please enter valid risk aversion method!')

    for start in range(length_data - window_size):
        end = start + window_size
        historical_data = monthly_return_entire_data.iloc[start:end]

        # Create bootstrap samples
        bootstrap_samples = create_bootstrap_samples(historical_data, num_simulations, num_samples, sample_size, method=simulation_method, df = df)
        
        # Optimize portfolio for bootstrap samples and store results
        if risk_seeking == True:
            result = optimize_portfolio_top_mean(bootstrap_samples, cutoff=alpha)
        else:
            result = optimize_portfolio_bottom_mean(bootstrap_samples, cutoff=alpha)
        
        all_results.append(result)
        risk_attitudes.append(risk_seeking)
        
        # update risk preference if dynamic
        if risk_aversion_method == 'most_recent_performance':
            optimal_weights = result[0]
            portfolio_performance = np.sum(np.dot(monthly_return_entire_data.iloc[end,:], optimal_weights))
            if portfolio_performance >= 1:
                risk_seeking = False
            else:
                risk_seeking = True
        
        elif risk_aversion_method == 'hidden_markov_model':
            state = hmm_regimes.iloc[start,0]
            if state == 'bear':
                risk_seeking = False
            else:
                risk_seeking = True
            
        print(f"Completed: {start}")

    return all_results, risk_attitudes

# Define Parameters
# all_historical_data = monthly_return.iloc[:, :9]
# num_simulations = 500
# num_samples = 20
# sample_size = 20
# simulation_method = 'multivariate_normal'
# df = 5
# window_size = 36
# alpha = 0.2
# risk_aversion_method = 'hidden_markov_model' # or 'hidden_markov_model'

# Example: Usage
# all_result, risk_attitudes = rolling_portfolio_optimization_proposed(
#     monthly_return_entire_data=all_historical_data,
#     num_simulations=num_simulations,
#     num_samples=num_samples,
#     sample_size=sample_size,
#     simulation_method=simulation_method,
#     window_size=window_size,
#     alpha=alpha,
#     risk_aversion_method = risk_aversion_method
# )

# Example: Output Results
# for weight, top_mean in all_result:
#     if weight is not None:
#         result = [round(num, 2) for num in weight]
#         print(result)

# Example: Validate Results
# historical_data = monthly_return.iloc[1:37, :9]  # Adjust column slicing as needed
# bootstrap_samples = create_bootstrap_samples(historical_data, num_simulations, num_samples, sample_size, method=simulation_method)
# result = optimize_portfolio_top_mean(bootstrap_samples, cutoff=alpha)
# print(result)

# wt = result[0]

# all_sample_top_mean = []
# for sample in bootstrap_samples:
#     all_sample_top_mean.append(np.mean(np.sort(np.array(sample @ wt))[-4:]))
    
# all_sample_top_mean
# np.sum(all_sample_top_mean)
# result[1]

# test
# column_name = np.array(monthly_return.iloc[:,:9].columns)
# all_weights = pd.DataFrame(columns = column_name)

# for i in range(264):
#     all_weights.loc[i+1] = all_result[i][0]
    
# test_return = monthly_return.iloc[36:,:9]

# all_weights.reset_index(drop = True, inplace=True)
# test_return.reset_index(drop = True, inplace=True)

# test_performance = all_weights * test_return
# monthly_performance = test_performance.sum(axis=1) + 1
# monthly_performance.cumprod()