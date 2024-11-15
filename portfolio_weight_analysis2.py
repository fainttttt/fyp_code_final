import os
import pandas as pd
import numpy as np


# obtain hmm regimes
relative_file_path = os.path.join(os.getcwd(), 'data_clean', '01_HMM_Regimes_BullBear.csv')
hmm_regimes = pd.read_csv(relative_file_path)


# obtain portfolio weight
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


# obtain lookup table
relative_file_path = os.path.join(os.getcwd(), 'data_clean', 'lookup_table.csv')
combined_df = pd.read_csv(relative_file_path)


filtered_df = combined_df[(combined_df['Risk'] == 'risk_averse') | (combined_df['Risk'] == 'risk_seeking')]
min_indicator = []
max_indicator = []

for lookup in filtered_df['lookup']:
    weight = portfolio_weight_obtained[lookup]  # Assuming this returns a dataframe for each lookup
    
    min_weight_count = 0
    max_weight_count = 0
    
    # Iterate over rows in the weight dataframe
    for index, row in weight.iterrows():
        zero_count = (row == 0).sum()  # Count how many elements are zero
        min_weight_count += zero_count
        max_count = (row == 0.3).sum()  # Count how many elements are 0.3
        max_weight_count += max_count
    
    # Append the counts to the respective lists
    min_indicator.append(min_weight_count)
    max_indicator.append(max_weight_count)

# Add the new columns to the combined_df dataframe
filtered_df['min_indicator'] = min_indicator
filtered_df['max_indicator'] = max_indicator
filtered_df['sum'] = combined_df['min_indicator'] + combined_df['max_indicator']
filtered_df.sort_values(['sum'])


pivot_df = filtered_df.pivot_table(index=['Method', 'Distribution'], columns='Risk', values='max_indicator', aggfunc='first')

# Reset column names and flatten the multi-index
pivot_df.columns = ['risk_averse', 'risk_seeking']
pivot_df.reset_index(inplace=True)

# The resulting dataframe with two columns for 'risk_averse' and 'risk_seeking'
print(pivot_df)