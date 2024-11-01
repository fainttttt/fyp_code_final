import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

os.chdir('/Users/fabian/Desktop/FYP/FYP_Code/code')
current_dir = os.getcwd()

relative_file_path = os.path.join(current_dir, 'data_clean', 'table_of_results.csv')
table_of_results = pd.read_csv(relative_file_path)

# Define the order of columns to keep
columns_to_keep = [
    'Method', 'Distribution', 'Risk Preference', 'Sharpe Ratio',
    'Portfolio Max Drawdown', 'Transaction Cost', 'Final Investment Amount',
    '5th Percentile (%)', '50th Percentile (%)', '95th Percentile (%)'
]

# Reorder and keep the specified columns
filter_table = table_of_results[columns_to_keep]
filter_table

# discussion on distribution
normal = filter_table[filter_table['Distribution'] == 'normal']
t = filter_table[filter_table['Distribution'] == 't']
skew_normal = filter_table[filter_table['Distribution'] == 'skew_normal']

# compare sharpe ratio
normal_sharpe = normal['Sharpe Ratio'].values
t_sharpe = t['Sharpe Ratio'].values
skew_normal_sharpe = skew_normal['Sharpe Ratio'].values

# compare portfolio max drawdown
normal_maxdd = normal['Portfolio Max Drawdown'].values
t_maxdd = t['Portfolio Max Drawdown'].values
skew_maxdd = skew_normal['Portfolio Max Drawdown'].values

np.mean(t_maxdd - normal_maxdd)
np.mean(skew_maxdd - normal_maxdd)


# discussion on risk preference -- Final Investment Amount
filter_columns = [
    'Method', 'Distribution', 'Risk Preference', 'Final Investment Amount'
]

fia = table_of_results[filter_columns]
fia = fia[fia['Risk Preference'].isin(['risk_averse', 'risk_seeking'])]
fia.columns.tolist()

# Create a pivot table for Transaction Cost
fia_table = fia.pivot_table(
    index=['Method', 'Distribution'], 
    columns='Risk Preference', 
    values='Final Investment Amount'
)

# discussion on risk preference -- Transaction Cost
filter_columns = [
    'Method', 'Distribution', 'Risk Preference', 'Transaction Cost'
]

tc = table_of_results[filter_columns]
tc_table = tc.pivot_table(
    index=['Method', 'Distribution'], 
    columns='Risk Preference', 
    values='Transaction Cost'
)

# discussion on risk preference -- Portfolio Max Drawdown
filter_columns = [
    'Method', 'Distribution', 'Risk Preference', 'Portfolio Max Drawdown'
]

maxdd = table_of_results[filter_columns]
maxdd_table = maxdd.pivot_table(
    index=['Method', 'Distribution'], 
    columns='Risk Preference', 
    values='Portfolio Max Drawdown'
)

# discussion on risk preference -- Transaction Cost
filter_columns = [
    'Method', 'Distribution', 'Risk Preference', 'Mean', 'SD', 'Sharpe Ratio',
    'Portfolio Max Drawdown', 'Transaction Cost', 'Final Investment Amount',
    '5th Percentile (%)', '50th Percentile (%)', '95th Percentile (%)'
]

sharpe_ratio = table_of_results.pivot_table(
    index=['Distribution', 'Risk Preference'], 
    columns='Method', 
    values='Sharpe Ratio'
)
sharpe_ratio

mpt = table_of_results[table_of_results['Method'] == 'mpt']
proposed = table_of_results[table_of_results['Method'] == 'proposed']

filter_columns = [
    'Distribution', 'Risk Preference', '5th Percentile (%)', '50th Percentile (%)', '95th Percentile (%)'
]

mpt_percentile = mpt[filter_columns]
proposed_percentile = proposed[filter_columns]

mpt_percentile
proposed_percentile