import numpy as np
import gurobipy as gp
from gurobipy import GRB, Model, quicksum
import os
import pandas as pd
import random
import time

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

def calculate_top_mean(model, bootstrap_sample, weights, cutoff=0.05):

    n_samples, num_assets = bootstrap_sample.shape

    # Calculate portfolio returns for each observation by multiplying weights with asset returns
    portfolio_returns = [
        quicksum(bootstrap_sample.iloc[i, j] * weights[j] for j in range(num_assets)) 
        for i in range(n_samples)
    ]

    # Binary variables to represent top percentile selection
    z = model.addVars(n_samples, vtype='B', name="z")  # Binary variables
    top_returns = model.addVars(n_samples, lb=0, name="top_returns")  # Auxiliary variable for top returns

    # Determine how many top percentile returns to select
    threshold_index = int(n_samples * cutoff)

    # Constraints to set the top returns based on selection
    for i in range(n_samples):
        model.addConstr(top_returns[i] == portfolio_returns[i] * z[i], name=f"top_return_{i}")

    # Ensure only the top percentile number of returns is selected
    model.addConstr(quicksum(z[i] for i in range(n_samples)) == threshold_index, name="percentile_selection")

    # Calculate the top mean by averaging the selected top percentile returns
    top_mean = (1 / threshold_index) * quicksum(top_returns[i] for i in range(n_samples))

    return top_mean

def optimize_portfolio_top_mean(all_bootstrap_samples, cutoff=0.05):
    num_assets = all_bootstrap_samples[0].shape[1]  # Number of assets (9 in your case)
    
    model = gp.Model("portfolio_optimization_top_mean")
    model.setParam('OutputFlag', 0)  # Suppress output
    
    # Define decision variables: weights for each asset, constrained to be non-negative (no shorting)
    weights = model.addVars(num_assets, lb=0, ub=0.3, name="weights")
    
    conditional_means = []
    
    # Loop through each bootstrap sample
    for sample in all_bootstrap_samples:
        top_mean = model.addVar(name="top_mean")  # Auxiliary variable for top mean return
        conditional_means.append(top_mean)
        
        # Add a constraint to calculate the top mean for each sample
        model.addConstr(top_mean == calculate_top_mean(model, sample, weights, cutoff), name=f"top_mean_{len(conditional_means) - 1}")
        
    # Set the objective to maximize the sum of conditional means across all bootstrap samples
    model.setObjective(quicksum(conditional_means), GRB.MAXIMIZE)
    
    # Ensure the sum of the weights equals 1 (fully invested portfolio)
    model.addConstr(quicksum(weights[i] for i in range(num_assets)) == 1, name="sum_weights")
    
    # Optimize the model
    start = time.time()
    model.optimize()
    end = time.time()
    duration = end - start
    print('Optimization completed in ' + str(duration) + ' seconds')
    
    optimal_weights = None
    sum_of_all_optimal_top_mean = None

    # Check if optimization was successful
    if model.status == GRB.OPTIMAL:
        optimal_weights = [weights[i].x for i in range(num_assets)]  # Extract the optimal weights
        sum_of_all_optimal_top_mean = model.objVal  # The value of the objective function (sum of maximum top mean return)
    else:
        model.computeIIS()
        model.write("infeasible_model.ilp")  # Save conflicting constraints for inspection if optimization fails
        print("Optimization failed with status code:", model.status)

    return optimal_weights, sum_of_all_optimal_top_mean

def rolling_portfolio_optimization_proposed(monthly_return_entire_data, 
                                            num_simulations=500, num_samples=100, sample_size=20, simulation_method='multivariate_normal',
                                            window_size=36, alpha=0.1):
    
    length_data = monthly_return_entire_data.shape[0]
    num_assets = monthly_return_entire_data.shape[1]
    all_results = []

    for start in range(length_data - window_size):
        end = start + window_size
        historical_data = monthly_return_entire_data.iloc[start:end, :num_assets]
        
        # Create bootstrap samples
        bootstrap_samples = create_bootstrap_samples(historical_data, num_simulations, num_samples, sample_size, method=simulation_method)
        
        # Optimize portfolio for bootstrap samples and store results
        all_results.append(optimize_portfolio_top_mean(bootstrap_samples, cutoff=alpha))

        print(f"Completed: {start}")

    return all_results

# Define Parameters
all_historical_data = monthly_return.iloc[:50, :9]  # Adjust column slicing as needed
num_simulations = 500
num_samples = 20 # 100
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

for weight, top_mean in all_result:
    result = [round(num,2) for num in weight]
    print(result)
### Model Validation!!! ###

# num_assets = 9
# initial_weights = np.ones(num_assets) / num_assets # equal weights for initialization
# historical_data = monthly_return.iloc[:36, :9]  # Adjust column slicing as needed
# bootstrap_samples = create_bootstrap_samples(historical_data, num_simulations, num_samples, sample_size, method=simulation_method)
# sample = bootstrap_samples[0]

# percentile = 1 - alpha
# threshold_index = int(len(sample)*percentile)
# np.mean(np.sort(np.dot(sample, initial_weights))[threshold_index:])
# calculate_top_mean(sample,initial_weights,alpha)

# # first iteration
# wt = all_result[0][0]

# all_historical_data = monthly_return.iloc[:37, :9]  # Adjust column slicing as needed
# bootstrap_samples = create_bootstrap_samples(all_historical_data, num_simulations, num_samples, sample_size, method=simulation_method)

# all_sample_top_mean = []
# for sample in bootstrap_samples:
#     all_sample_top_mean.append(np.mean(np.sort(np.array(sample @ wt))[-4:]))
    
# np.sum(all_sample_top_mean)
# all_result[0][1]

# # second iteration
# wt = all_result[1][0]

# all_historical_data = monthly_return.iloc[1:38, :9]  # Adjust column slicing as needed
# bootstrap_samples = create_bootstrap_samples(all_historical_data, num_simulations, num_samples, sample_size, method=simulation_method)

# all_sample_top_mean = []
# for sample in bootstrap_samples:
#     all_sample_top_mean.append(np.mean(np.sort(np.array(sample @ wt))[-4:]))
    
# np.sum(all_sample_top_mean)
# all_result[1][1]


historical_data = monthly_return.iloc[:36, :9]  # Adjust column slicing as needed
bootstrap_samples = create_bootstrap_samples(historical_data, num_simulations, num_samples, sample_size, method=simulation_method)
result = optimize_portfolio_top_mean(bootstrap_samples, cutoff=alpha)
print(result)

# first iteration
wt = result[0]

all_sample_top_mean = []
for sample in bootstrap_samples:
    all_sample_top_mean.append(np.mean(np.sort(np.array(sample @ wt))[-4:]))
    
all_sample_top_mean
np.sum(all_sample_top_mean)
result[1]