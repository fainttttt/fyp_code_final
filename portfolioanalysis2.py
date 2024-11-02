import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('/Users/fabian/Desktop/FYP/FYP_Code/code')
current_dir = os.getcwd()

relative_file_path = os.path.join(current_dir, 'data_clean', 'final_compiled_results.csv')
my_results = pd.read_csv(relative_file_path)
my_results.head()


# 6.1 Discussion on Distribution 

# generate pivot table for presentation --- max dd #
my_pivot = my_results.pivot_table(index=['Method', 'Risk'], 
                                 columns='Distribution', 
                                 values='max_dd').reset_index()

my_pivot.reset_index(drop = True, inplace = True)
my_pivot['skew_normal_dff'] = my_pivot['skew_normal'] - my_pivot['normal']
my_pivot['t_dff'] = my_pivot['t'] - my_pivot['normal']
print(my_pivot)


# 6.2 Discussion on Risk

# generate pivot table for presentation --- tc #
my_pivot = my_results.pivot_table(index=['Method', 'Distribution'], 
                                 columns='Risk', 
                                 values='tc').reset_index()
my_pivot.reset_index(drop = True, inplace = True)
print(my_pivot)


# generate pivot table for presentation --- tc #
my_pivot = my_results.pivot_table(index=['Method', 'Distribution'], 
                                 columns='Risk', 
                                 values='final_investment_amount').reset_index()
my_pivot.reset_index(drop = True, inplace = True)
print(my_pivot)


# 6.3 Discussion on Method
# generate pivot table for presentation --- sharpe ratio #
my_pivot = my_results.pivot_table(index=['Distribution', 'Risk'], 
                                 columns='Method', 
                                 values='sharpe_ratio').reset_index()
my_pivot.reset_index(drop = True, inplace = True)
# my_pivot['bias_cost'] = my_pivot['mpt'] - my_pivot['proposed']
print(my_pivot)

# generate pivot table for presentation --- 5th percentile #
my_pivot = my_results.pivot_table(index=['Distribution', 'Risk'], 
                                 columns='Method', 
                                 values='5th_percentile').reset_index()
my_pivot.reset_index(drop = True, inplace = True)
my_pivot['diff'] = my_pivot['mpt'] - my_pivot['proposed']
print(my_pivot)

# generate pivot table for presentation --- 95th percentile #
my_pivot = my_results.pivot_table(index=['Distribution', 'Risk'], 
                                 columns='Method', 
                                 values='95th_percentile').reset_index()
my_pivot.reset_index(drop = True, inplace = True)
my_pivot['diff'] = my_pivot['mpt'] - my_pivot['proposed']
print(my_pivot)

# generate pivot table for presentation --- final investment amount #
my_pivot = my_results.pivot_table(index=['Distribution', 'Risk'], 
                                 columns='Method', 
                                 values='final_investment_amount').reset_index()
my_pivot.reset_index(drop = True, inplace = True)
my_pivot['bias_cost'] = (my_pivot['proposed'] / my_pivot['mpt']) - 1
print(my_pivot)


# generate pivot table for presentation --- final investment amount #
my_pivot = my_results.pivot_table(index=['Method', 'Distribution'], 
                                 columns='Risk', 
                                 values='final_investment_amount').reset_index()
my_pivot.reset_index(drop = True, inplace = True)
my_pivot['bias_cost_ra'] = (my_pivot['risk_averse'] / my_pivot['mrp']) - 1
my_pivot['bias_cost_rs'] = (my_pivot['risk_seeking'] / my_pivot['mrp']) - 1
print(my_pivot)



### Overview of Results ###
# # my_filter_portfolios = my_results[my_results['Risk'].isin(['risk_averse', 'risk_seeking'])]
my_filter_portfolios = my_results

# Create a scatter plot for Sharpe Ratio vs Max Drawdown
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    data = my_filter_portfolios,
    x = 'max_dd',
    y = 'sharpe_ratio',
    size = 'final_investment_amount',
    sizes = (100, 1000),  # Adjust size range
    hue = 'tc',
    palette = 'viridis',
    alpha = 0.6,
    legend = False
)

# # Add annotations
# for i in range(my_filter_portfolios.shape[0]):
#     plt.text(my_filter_portfolios['sharpe_ratio'].iloc[i], my_filter_portfolios['max_dd'].iloc[i], 
#              my_filter_portfolios['lookup'].iloc[i], fontsize=8)

# Customize the plot
plt.title('Portfolio Performance: Sharpe Ratio vs Max Drawdown', fontsize=16)
plt.xlabel('Max Drawdown', fontsize=14)
plt.ylabel('Sharpe Ratio', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.colorbar(scatter.collections[0], label='Transaction Costs')

# Show the plot
plt.savefig('Overview_of_Results.png', dpi=300, bbox_inches='tight')
plt.show()


### Analysis of Distribution ###
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=my_filter_portfolios,
    x='max_dd', 
    y='sharpe_ratio', 
    hue='Distribution',
    style='Distribution',
    markers=['X'],  # Use crosses
    s=100  # Adjust size for visibility
)

# Customizing the plot
plt.title('Sharpe Ratio vs. Max Drawdown by Distribution')
plt.xlabel('Max Drawdown')
plt.ylabel('Sharpe Ratio')
plt.legend(title='Distribution', bbox_to_anchor=(0.8, 1), loc='upper left')
plt.savefig('Analysis_of_Distribution.png', dpi=300, bbox_inches='tight')
plt.show()


### Analysis of Risk Preferences ###
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=my_filter_portfolios,
    x='tc', 
    y='final_investment_amount', 
    hue='Risk',
    style='Risk',
    markers=['X'],  # Use crosses
    s=100  # Adjust size for visibility
)

# Customizing the plot
plt.title('Final Investment Amount vs. Transaction Costs by Risk')
plt.xlabel('Transaction Cost')
plt.ylabel('Final Investment Amount')
plt.legend(title='Risk', bbox_to_anchor=(0.8, 1), loc='upper left')
plt.savefig('Analysis_of_Risk.png', dpi=300, bbox_inches='tight')
plt.show()



### Analysis of Methods -- Sharpe Ratio ###

# generate pivot table for presentation --- sharpe ratio #
my_pivot = my_results.pivot_table(index=['Distribution','Risk'], 
                                 columns='Method', 
                                 values='sharpe_ratio').reset_index()
my_pivot.reset_index(drop = True, inplace = True)
# my_pivot['bias_cost'] = my_pivot['mpt'] - my_pivot['proposed']
print(my_pivot)

# Set the figure size
plt.figure(figsize=(10, 6))

# Plot the values for Method 1
plt.plot(my_pivot.index, my_pivot['mpt'], marker='o', label='mpt', color='blue')

# Plot the values for Method 2
plt.plot(my_pivot.index, my_pivot['proposed'], marker='s', label='proposed', color='orange')

# Draw lines connecting corresponding values
for i in range(len(my_pivot)):
    plt.plot([i, i], [my_pivot['mpt'].iloc[i], my_pivot['proposed'].iloc[i]], color='gray', linestyle='--', alpha=0.5)

# Adding titles and labels
plt.title('Comparison of MPT and Proposed Methods')
plt.xlabel('Portfolio Combinations - Distribution and Risk')
plt.ylabel('Sharpe Ratio')
plt.xticks(range(len(my_pivot)), range(1, len(my_pivot) + 1))  # Optional: to show indices as 1, 2, 3, ...
plt.legend()

# Show the plot
plt.tight_layout()
# plt.savefig('Analysis_of_Method_SR.png', dpi=300, bbox_inches='tight')
plt.show()


### Analysis of Methods -- Percentile ###

# my_filter_portfolios = my_results[['Method','Distribution','Risk','5th_percentile','95th_percentile']]
# my_filter_portfolios['lookup'] = my_filter_portfolios['Distribution'] + "_" + my_filter_portfolios['Risk']

import matplotlib.pyplot as plt
import pandas as pd

# Ensure my_results is defined and has the necessary columns
# Example DataFrame creation for demonstration (remove this if you already have my_results)
# my_results = pd.DataFrame({
#     'Method': ['Group1', 'Group2', 'Group1', 'Group2', 'Group1'],
#     '5th_percentile': [10, 15, 20, 25, 30],
#     '95th_percentile': [40, 45, 50, 55, 60]
# })

# Filter portfolios
my_filter_portfolios = my_results[['Method', '5th_percentile', '95th_percentile']]

# Map method groups to colors
color_map = {'mpt': 'blue', 'proposed': 'orange'}
my_filter_portfolios['Color'] = my_filter_portfolios['Method'].map(color_map)

# Create a scatter plot
plt.figure(figsize=(8, 5))
scatter = plt.scatter(my_filter_portfolios['95th_percentile'], 
                      my_filter_portfolios['5th_percentile'], 
                      color=my_filter_portfolios['Color'], 
                      s=50,  # Adjust marker size here
                      marker='x')  # Use 'x' for cross markers

# Set labels and title
plt.xlabel('95th Percentile')
plt.ylabel('5th Percentile')
plt.title('Scatter Plot of Percentile Returns across Methods')
plt.savefig('Analysis_of_Method_Percentile_New.png', dpi=300, bbox_inches='tight')
plt.show()





# plt.figure(figsize=(10, 6))
# sns.scatterplot(
#     data = my_filter_portfolios,
#     x = '5th_percentile',
#     y='95th_percentile',
#     hue='Method',  # Color by group
#     # style='Method',  # Different markers for each group
#     s=100  # Marker size
# )

# # Linking pairs of points using the lookup column
# for name, group in my_filter_portfolios.groupby('lookup'):
#     if len(group) > 1:
#         # Extract the x and y values for the pairs
#         x_values = group['5th_percentile'].values
#         y_values = group['95th_percentile'].values
        
#         # Plot lines between points in the same group
#         plt.plot(
#             x_values,
#             y_values,
#             linestyle='-', 
#             color='grey', 
#             alpha=0.5
#         )

# plt.title('Scatter Plot of 95th_percentile vs. 5th_percentile')
# plt.xlabel('5th_percentile')
# plt.ylabel('95th_percentile')
# plt.legend(title='Method', bbox_to_anchor=(0.8, 1), loc='upper left')
# plt.tight_layout()
# plt.savefig('Analysis_of_Method_Percentile.png', dpi=300, bbox_inches='tight')
# plt.show()