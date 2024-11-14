import pandas as pd
import numpy as np
from scipy.optimize import minimize
from data_simulation import estimate_skew_normal_params
from data_simulation import simulate_return
from data_simulation import create_bootstrap_samples

import os
os.chdir('/Users/fabian/Desktop/FYP/FYP_Code/code')
# print("Current Working Directory:", os.getcwd())

current_dir = os.getcwd()

relative_file_path = os.path.join(os.getcwd(), 'data_clean', '9_Sectors_Ticker_Monthly_Returns.csv')
monthly_return = pd.read_csv(relative_file_path)

relative_file_path = os.path.join(os.getcwd(), 'data_clean', '01_HMM_Regimes_BullBear.csv')
hmm_regimes = pd.read_csv(relative_file_path)

risk_averse = 1
risk_seeking = -1 # may not be possible in mpt algorithm

# Using np.where to create a new array based on the conditions
hmm_regime_parameter = np.where(hmm_regimes.iloc[:, 0] == 'bull', risk_seeking, risk_averse)
hmm_regimes.iloc[:,0] = hmm_regime_parameter
hmm_regimes.head()

# Function to simulate returns based on the specified method
def calculate_utility(train_data, weights_to_try, chosen_risk_aversion_parameter): # apply weights on bootstrap samples of size 20
    
    portfolio_returns = np.dot(train_data, weights_to_try) # apply weight to bootstrap samples to obtain sequence of portfolio return
    return_mean = portfolio_returns.mean()
    return_std = np.sqrt(portfolio_returns.var())
    utility = return_mean - (chosen_risk_aversion_parameter * return_std)
    
    return utility  

def objective_function(weights, bootstrap_samples, chosen_risk_aversion_parameter):
    total_utility = 0
    for sample in bootstrap_samples:
        total_utility += calculate_utility(sample, weights, chosen_risk_aversion_parameter) # cumulatively add utility across all bootstrap_samples
    average_utility = total_utility / len(bootstrap_samples)
    return -average_utility # Negative because we will minimize this value

def rolling_portfolio_optimization_mpt(monthly_return_entire_data, 
                                       num_simulations=500, num_samples=100, sample_size=20, simulation_method = 'multivariate_normal', df = 5,
                                       window_size=36, risk_aversion_method = 'most_recent_performance'):
    
    column_names = monthly_return_entire_data.columns.tolist()
    
    length_data = monthly_return_entire_data.shape[0] # number of time periods
    num_assets = monthly_return_entire_data.shape[1] # number of assets

    all_optimal_weights = [] # to store optimal weights in each rolling window
    utility_list = [] # to store optimal utility in each rolling window

    # Define constraints and bounds for optimization
    initial_weights = np.ones(num_assets) / num_assets # equal weights for initialization
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Sum of weights = 1
    bounds = [(0, 0.3) for _ in range(num_assets)]  # No shorting, weights between 0 and 0.3
    
    if risk_aversion_method == 'most_recent_performance': # set first period as risk averse
        risk_aversion = risk_averse
    elif risk_aversion_method == 'hidden_markov_model':
        risk_aversion = hmm_regimes.iloc[0,0]
    elif risk_aversion_method == 'all_risk_seeking':
        risk_aversion = risk_seeking
    elif risk_aversion_method == 'all_risk_averse':
        risk_aversion = risk_averse
    else:
        print('Please enter valid risk aversion method!')
        
    for start in range(length_data - window_size):
        
        end = start + window_size
        historical_data = monthly_return_entire_data.iloc[start:end, :] # select the training data for rolling window
        
        bootstrap_samples = create_bootstrap_samples(historical_data, num_simulations, num_samples, sample_size, method = simulation_method, df = 5) # prepare bootstrap samples

        # Minimize the objective function
        result = minimize(objective_function, initial_weights, args=(bootstrap_samples, risk_aversion), constraints=constraints, bounds=bounds, method='SLSQP')
        
        optimal_weights = result.x
        all_optimal_weights.append(optimal_weights)

        optimized_utility = -objective_function(optimal_weights, bootstrap_samples, risk_aversion) # utility derived from optimized weight
        ew_utility = -objective_function(initial_weights, bootstrap_samples, risk_aversion) # utility derived from equal weight
        utility_pair = (optimized_utility, ew_utility)
        utility_list.append(utility_pair)

        # update risk aversion parameter if dynamic
        if risk_aversion_method == 'most_recent_performance':
            portfolio_performance = np.sum(np.dot(monthly_return_entire_data.iloc[end,:], optimal_weights))
            if portfolio_performance >= 1:
                risk_aversion = risk_averse
            else:
                risk_aversion = risk_seeking
        elif risk_aversion_method == 'hidden_markov_model':
            risk_aversion = hmm_regimes.iloc[start,0]
        
        print(f"Completed: {start}")
        
    return pd.DataFrame(all_optimal_weights, columns=column_names), utility_list

# Define Parameters
# all_historical_data = monthly_return.iloc[:,:9]
# num_simulations = 500
# num_samples = 100
# sample_size = 20
# simulation_method = 'multivariate_normal'
# df = 5
# window_size = 36
# risk_aversion_method = 'hidden_markov_model'

# # Usage Example
# optimal_weights_df, utility_results = rolling_portfolio_optimization_mpt(monthly_return_entire_data = all_historical_data,
#                                                                          num_simulations = num_simulations,
#                                                                          num_samples = num_samples,
#                                                                          sample_size = sample_size,
#                                                                          simulation_method = simulation_method,
#                                                                          window_size = window_size,
#                                                                          risk_aversion_method = risk_aversion_method)

# utility_results is evaluated based on bootstrap samples - we need multiple return values to determine standard deviation then utility under mpt model
# risk aversion method can be (1) most recent performance or (2) hidden markov model
# simulation method can be (1) multivariate normal or (2) 

# we set risk averse = 1, risk seeking = -1
# setting risk seeking = 1 leads to full allocation to one asset which is problematic
# hence we impose upper limit on max allocation to any one asset as 0.3

# optimal_weights_df.round(5).head(30)