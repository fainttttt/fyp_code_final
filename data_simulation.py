import os
os.chdir('/Users/fabian/Desktop/FYP/FYP_Code/code')
# print("Current Working Directory:", os.getcwd())

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
# norm_data = simulate_return(historical_data, num_simulations, method = 'multivariate_normal', df = df)
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


mae_norm = []
mse_norm = []
rmse_norm = []

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
    mae_norm.append(sum(np.array(mae_list)))
    mse_norm.append(sum(np.array(mse_list)))
    rmse_norm.append(sum(np.array(rmse_list)))

mae_t = []
mse_t = []
rmse_t = []

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
    mae_t.append(sum(np.array(mae_list)))
    mse_t.append(sum(np.array(mse_list)))
    rmse_t.append(sum(np.array(rmse_list)))

mae_sn = []
mse_sn = []
rmse_sn = []

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
    mae_sn.append(sum(np.array(mae_list)))
    mse_sn.append(sum(np.array(mse_list)))
    rmse_sn.append(sum(np.array(rmse_list)))

folder_path = 'data_clean'

model_valiation_df = pd.DataFrame({
    'mae_norm': mae_norm,
    'mae_t': mae_t,
    'mae_sn': mae_sn,
    'mse_norm': mse_norm,
    'mse_t': mse_t,
    'mse_sn': mse_sn,
    'rmse_norm': rmse_norm,
    'rmse_t': rmse_t,
    'rmse_sn': rmse_sn
})

csv_file = os.path.join(folder_path,"simulation_model_validation.csv")
model_valiation_df.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")