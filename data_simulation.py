import os
os.chdir('/Users/fabian/Desktop/FYP/FYP_Code/code')
print("Current Working Directory:", os.getcwd())

import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
from scipy.stats import skewnorm, skew, t

current_dir = os.getcwd()

relative_file_path = os.path.join(current_dir, 'data_clean', '9_Sectors_Ticker_Monthly_Returns.csv')
monthly_return = pd.read_csv(relative_file_path)

# Function to estimate parameters for the skew normal distribution from historical data
def estimate_skew_normal_params(historical_data):
    params = {}
    for asset in historical_data.columns:
        # Calculate mean (location)
        location = historical_data[asset].mean()
        
        # Calculate standard deviation (scale)
        scale = historical_data[asset].std()
        
        # Calculate skewness
        skewness = skew(historical_data[asset])
        
        params[asset] = (location, scale, skewness)
    
    return params

def simulate_return(historical_data, num_simulations, method='multivariate_normal', df = 5):
    np.random.seed(123)
    column_names = historical_data.columns.tolist()

    if method == 'multivariate_normal':
        mean_estimate = historical_data.mean().values
        cov_estimate = historical_data.cov().values
        simulated_returns = pd.DataFrame(
            np.random.multivariate_normal(mean_estimate, cov_estimate, num_simulations),
            columns=column_names
        )
    
    elif method == 'multivariate_t':
        df = 5  # Degrees of freedom
        mean_estimate = historical_data.mean().values
        cov_estimate = historical_data.cov().values
        L = np.linalg.cholesky(cov_estimate)
        z = np.random.standard_t(df, size=(num_simulations, len(column_names)))
        simulated_returns = pd.DataFrame(
            mean_estimate + z @ L.T,
            columns=column_names
        )

    elif method == 'skew_normal':
        params = estimate_skew_normal_params(historical_data)
        if params is None:
            raise ValueError("Unable to estimate skew normal params from historical returns.")

        # Create an empty DataFrame to store simulated returns
        simulated_returns = pd.DataFrame(index=range(num_simulations), columns=column_names)

        # Simulate returns for each asset using the parameters
        for asset in column_names:
            location, scale, skewness = params[asset]
            simulated_returns[asset] = skewnorm.rvs(a=skewness, loc=location, scale=scale, size=num_simulations)

    return simulated_returns

def create_bootstrap_samples(historical_data, num_simulations, num_samples, sample_size, method = 'multivariate_normal', df = 5):
    
    simulated_return = simulate_return(historical_data, num_simulations, method, df)

    bootstrap_samples = []
    for i in range(num_samples):
        np.random.seed(i)
        bootstrap_df = simulated_return.sample(sample_size,replace=True)
        bootstrap_samples.append(bootstrap_df)
        
    return(bootstrap_samples)

historical_data = monthly_return.iloc[:36,:9]
num_simulations = 500
num_samples = 100
sample_size = 20
simulation_method = 'skew_normal' # multivariate_t' or 'multivariate_normal'

# parameter required for t distribution
df = 5 # specify tail heaviness

# # parameter required for skew normal distribution

# t_data = simulate_return(historical_data, num_simulations, method = 'multivariate_t', df = df)
norm_data = simulate_return(historical_data, num_simulations, method = 'multivariate_normal', df = df)
# skew_norm_data = simulate_return(historical_data, num_simulations, method = 'skew_normal', df = df)

# cba = create_bootstrap_samples(historical_data, num_simulations, num_samples, sample_size, method = 'multivariate_t', df = df)

# # examine simulated data
# def calculate_statistics_across_columns(dataframe):
#     # Initialize a dictionary to store the results
#     statistics = {
#         'mean': {},
#         'variance': {},
#         'skewness': {}
#     }
    
#     # Calculate statistics for each column
#     for column in dataframe.columns:
#         statistics['mean'][column] = dataframe[column].mean()
#         statistics['variance'][column] = dataframe[column].var()  # Use .var(ddof=0) for population variance
#         statistics['skewness'][column] = skew(dataframe[column])

#     return statistics

# t_stats = calculate_statistics_across_columns(t_data)
# norm_stats = calculate_statistics_across_columns(norm_data)
# skew_norm_stats = calculate_statistics_across_columns(skew_norm_data)

# stats_name = ['mean','variance','skenwess']
# column_name = historical_data.columns.to_list()

# for stats in stats_name:
#     for column in column_name:
#         print(f'{stats} for {column} in t distribution equals {t_stats[str(stats)][str(column)]}')
#         print(f'{stats} for {column} in norm distribution equals {norm_stats[str(stats)][str(column)]}')
#         print(f'{stats} for {column} in skew_norm distribution equals {skew_norm_stats[str(stats)][str(column)]}')
#         print('\n')

mae_all = []
mse_all = []
rmse_all = []

for start in range(264):
    test_data = monthly_return.iloc[start:start+36,:9]
    norm_data = simulate_return(test_data, num_simulations, method = 'multivariate_normal', df = df)
    true_data = monthly_return.iloc[start+36,:9]

    # Initialize lists to store errors
    mae_list = []
    mse_list = []
    rmse_list = []

    # Calculate errors for each row
    for ix in range(num_simulations):
        row = norm_data.loc[ix]
        # Calculate errors
        mae = np.mean(np.abs(row - true_data))
        mse = np.mean((row - true_data) ** 2)
        rmse = np.sqrt(mse)

        # Append results to the lists
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)

    # Convert lists to NumPy arrays for easier manipulation
    mae_all.append(sum(np.array(mae_list)))
    mse_all.append(sum(np.array(mse_list)))
    rmse_all.append(sum(np.array(rmse_list)))
    
mae_norm = mae_all
mse_norm = mse_all
rmse_norm = rmse_all

mae_all = []
mse_all = []
rmse_all = []

for start in range(264):
    test_data = monthly_return.iloc[start:start+36,:9]
    norm_data = simulate_return(test_data, num_simulations, method = 'multivariate_t', df = df)
    true_data = monthly_return.iloc[start+36,:9]

    # Initialize lists to store errors
    mae_list = []
    mse_list = []
    rmse_list = []

    # Calculate errors for each row
    for ix in range(num_simulations):
        row = norm_data.loc[ix]
        # Calculate errors
        mae = np.mean(np.abs(row - true_data))
        mse = np.mean((row - true_data) ** 2)
        rmse = np.sqrt(mse)

        # Append results to the lists
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)

    # Convert lists to NumPy arrays for easier manipulation
    mae_all.append(sum(np.array(mae_list)))
    mse_all.append(sum(np.array(mse_list)))
    rmse_all.append(sum(np.array(rmse_list)))
    
mae_t = mae_all
mse_t = mse_all
rmse_t = rmse_all

mae_all = []
mse_all = []
rmse_all = []

for start in range(264):
    test_data = monthly_return.iloc[start:start+36,:9]
    norm_data = simulate_return(test_data, num_simulations, method = 'skew_normal', df = df)
    true_data = monthly_return.iloc[start+36,:9]

    # Initialize lists to store errors
    mae_list = []
    mse_list = []
    rmse_list = []

    # Calculate errors for each row
    for ix in range(num_simulations):
        row = norm_data.loc[ix]
        # Calculate errors
        mae = np.mean(np.abs(row - true_data))
        mse = np.mean((row - true_data) ** 2)
        rmse = np.sqrt(mse)

        # Append results to the lists
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)

    # Convert lists to NumPy arrays for easier manipulation
    mae_all.append(sum(np.array(mae_list)))
    mse_all.append(sum(np.array(mse_list)))
    rmse_all.append(sum(np.array(rmse_list)))
    
mae_sn = mae_all
mse_sn = mse_all
rmse_sn = rmse_all

mae_norm
mae_t

import matplotlib.pyplot as plt
import numpy as np

# Assuming the x-axis is the same for all three lists
x = np.arange(len(mae_norm))  # Create an x-axis based on the length of the lists

# Create the plot
plt.figure(figsize=(14, 7))  # Set a larger figure size

# Plot each list with different styles
plt.plot(x, rmse_norm, label='Normal', marker='', linestyle='-', alpha=0.6)  # Line plot
plt.plot(x, rmse_t, label='T', marker='', linestyle='--', alpha=0.6)  # Dashed line plot
plt.plot(x, rmse_sn, label='Skew Normal', marker='', linestyle=':', alpha=0.6)  # Dotted line plot

# Reduce the number of x-ticks for better readability
tick_indices = np.arange(0, len(mae_norm), 20)  # Show every 20th tick
plt.xticks(tick_indices, tick_indices)

# Add titles and labels
plt.title('Plot of Three Long Lists')
plt.xlabel('Index')
plt.ylabel('RMSE')

# Add a legend
plt.legend()

# Show the grid for better readability
plt.grid(True)

# Show the plot
plt.show()