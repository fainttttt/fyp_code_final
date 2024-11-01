import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import re

os.chdir('/Users/fabian/Desktop/FYP/FYP_Code/code')
current_dir = os.getcwd()

relative_file_path = os.path.join(current_dir, 'data_clean', '9_Sectors_Ticker_Monthly_Returns.csv')
monthly_return = pd.read_csv(relative_file_path)
test_return = monthly_return.iloc[36:,:9]
test_return.reset_index(drop = True, inplace = True) ## filter for required monthly returns only

def read_portfolio_weights(folder_path):
    # Dictionary to store dataframes with filenames as keys
    csv_dict = {}

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a CSV
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            
            # Read the CSV file into a dataframe
            df = pd.read_csv(file_path)
            
            # Use the file name (without extension) as the key in the dictionary
            csv_dict[file_name[:-4]] = df
    
    return csv_dict

folder_path = '/Users/fabian/Desktop/FYP/FYP_Code/code/portfolio_weights'
portfolio_weight_obtained = read_portfolio_weights(folder_path)

# collect portfolio names into a python list
portfolio_names = []
for key,value in portfolio_weight_obtained.items():
    portfolio_names.append(key)
portfolio_names

# find number of maxed-out allocation
max_indicator = pd.DataFrame(columns = portfolio_names)
max_indicator.shape

for ix in range(len(portfolio_names)):
    portfolio_name = portfolio_names[ix]
    portfolio_weight = portfolio_weight_obtained[portfolio_name]
    bool_df = (portfolio_weight == 0.3).astype(int)
    vector = np.sum(bool_df,axis = 0).values
    max_indicator[portfolio_name] = vector

max_indicator.index = portfolio_weight_obtained[portfolio_names[0]].columns.tolist()
max_indicator = max_indicator.T
max_indicator['lookup'] = max_indicator.index
max_indicator

os.chdir('/Users/fabian/Desktop/FYP/FYP_Code/code')
current_dir = os.getcwd()
relative_file_path = os.path.join(current_dir, 'data_clean', 'lookup_table.csv')
lookup_df = pd.read_csv(relative_file_path)

max_indicator = pd.merge(max_indicator, lookup_df, on='lookup', how='inner')
max_indicator = max_indicator.sort_values(by=['Method', 'Distribution', 'Risk'])
max_indicator.reset_index(drop = True, inplace = True)

max_indicator['row_sum'] = max_indicator.iloc[:, :9].sum(axis=1)
print(max_indicator)

my_pivot = max_indicator.pivot_table(index=['Distribution', 'Risk'], 
                                     columns='Method', 
                                     values='row_sum').reset_index()
my_pivot.reset_index(drop = True, inplace = True)
print(my_pivot)