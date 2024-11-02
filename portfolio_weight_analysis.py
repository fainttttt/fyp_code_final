import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# find number of zero allocation
zero_indicator = pd.DataFrame(columns = portfolio_names)
zero_indicator.shape

for ix in range(len(portfolio_names)):
    portfolio_name = portfolio_names[ix]
    portfolio_weight = portfolio_weight_obtained[portfolio_name]
    bool_df = (portfolio_weight == 0).astype(int)
    vector = np.sum(bool_df,axis = 0).values
    zero_indicator[portfolio_name] = vector

zero_indicator.index = portfolio_weight_obtained[portfolio_names[0]].columns.tolist()
zero_indicator = zero_indicator.T
zero_indicator['lookup'] = zero_indicator.index
zero_indicator

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

zero_indicator = pd.merge(zero_indicator, lookup_df, on='lookup', how='inner')
zero_indicator = zero_indicator.sort_values(by=['Method', 'Distribution', 'Risk'])
zero_indicator.reset_index(drop = True, inplace = True)

max_indicator = pd.merge(max_indicator, lookup_df, on='lookup', how='inner')
max_indicator = max_indicator.sort_values(by=['Method', 'Distribution', 'Risk'])
max_indicator.reset_index(drop = True, inplace = True)

# number of times we hit diversification threshold across assets and methods
zero_filtered_data = zero_indicator.iloc[:, list(range(9)) + [10]]
zero_stats = zero_filtered_data.groupby('Method').mean()
zero_stats

# number of times we hit diversification threshold across assets and methods
max_filtered_data = max_indicator.iloc[:, list(range(9)) + [10]]
max_stats = max_filtered_data.groupby('Method').mean()
max_stats

max_stats_transpose = max_stats.T
zero_stats_transpose = zero_stats.T

# Plotting
plt.figure(figsize=(14, 7))

# Plotting Zero Hits
for column in zero_stats_transpose.columns:
    plt.plot(zero_stats_transpose.index, zero_stats_transpose['mpt'], marker='o', linestyle='-', 
             label=f'Zero Hits - MPT', color='blue')  # Zero Hits Method 1

for column in zero_stats_transpose.columns:
    plt.plot(zero_stats_transpose.index, zero_stats_transpose['proposed'], marker='x', linestyle='--', 
             label=f'Zero Hits - Proposed', color='blue')  # Zero Hits Method 2
    
plt.text(zero_stats_transpose.index[-1], zero_stats_transpose['mpt'].iloc[-1], ' Zero Hits - MPT', 
         verticalalignment='center', color='blue', fontsize=10)
plt.text(zero_stats_transpose.index[-1], zero_stats_transpose['proposed'].iloc[-1], ' Zero Hits - Proposed', 
         verticalalignment='center', color='blue', fontsize=10)

# Plotting Threshold Hits
for column in max_stats_transpose.columns:
    plt.plot(max_stats_transpose.index, max_stats_transpose['mpt'], marker='s', linestyle='-', 
             label=f'Threshold Hits - MPT', color='orange')  # Threshold Hits Method 1

for column in max_stats_transpose.columns:
    plt.plot(max_stats_transpose.index, max_stats_transpose['proposed'], marker='d', linestyle='--', 
             label=f'Threshold Hits - Proposed', color='orange')  # Threshold Hits Method 2
    
plt.text(max_stats_transpose.index[-1], max_stats_transpose['mpt'].iloc[-1], ' Threshold Hits - MPT', 
         verticalalignment='center', color='orange', fontsize=10)
plt.text(max_stats_transpose.index[-1], max_stats_transpose['proposed'].iloc[-1], ' Threshold Hits - Proposed', 
         verticalalignment='center', color='orange', fontsize=10)

plt.title('Comparison of Mean Hits for Different Methods across Sectors')
plt.xlabel('Sectors')
plt.ylabel('Number of Hits')
plt.grid(True)

# Adjust x-axis labels
plt.xticks(rotation=45, fontsize=10)  # Rotate and reduce font size
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig('Analysis_of_Portfolio_Weight.png', dpi=300, bbox_inches='tight')
plt.show()



# max_indicator['row_sum'] = max_indicator.iloc[:, :9].sum(axis=1)
# print(max_indicator)

# my_pivot = max_indicator.pivot_table(index=['Distribution', 'Risk'], 
#                                      columns='Method', 
#                                      values='row_sum').reset_index()
# my_pivot.reset_index(drop = True, inplace = True)
# print(my_pivot)