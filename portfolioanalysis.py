import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import re

os.chdir('/Users/fabian/Desktop/FYP/FYP_Code/code')
current_dir = os.getcwd()

# relative_file_path = os.path.join(current_dir, 'data_raw', 'risk_free_rates.csv')
# fed_rates = pd.read_csv(relative_file_path)
# fed_rates['DATE'] = pd.to_datetime(fed_rates['DATE'])
# filtered_dates = fed_rates[(fed_rates['DATE'].dt.year >= 2002) & (fed_rates['DATE'].dt.year <= 2023)]
# filtered_fed_rates = filtered_dates['FEDFUNDS'].values ## filter for required risk free rates only

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

### portfolio_weight_obtained: portfolio weight obtained across 25 variations ###
### test_return: test return to be evaluated ###

def extract_components(param_string):
    # Define regex patterns for methods, distributions, and risks
    methods = r'(mpt|proposed)'
    distributions = r'(normal|t|skew_normal)'
    risks = r'(risk_seeking|risk_averse|mrp|hmm)'
    
    # Create a combined regex pattern
    pattern = fr'^{methods}_{distributions}_{risks}$'
    
    # Match the pattern against the input string
    match = re.match(pattern, param_string)
    
    if match:
        return match.groups()
    else:
        raise ValueError("Invalid parameter string.")

portfolio_choices = list(portfolio_weight_obtained.keys())
portfolio_choices.remove('equal_weight')

methods = []
distributions = []
risks = []

for param_string in portfolio_choices:
    method, distribution, risk = extract_components(param_string)
    methods.append(method)
    distributions.append(distribution)
    risks.append(risk)
    # print(f"Method: {method}, Distribution: {distribution}, Risk: {risk}")
    
combined_df = pd.DataFrame({
    'lookup': portfolio_choices,
    'Method': methods,
    'Distribution': distributions,
    'Risk': risks
})

folder_path = '/Users/fabian/Desktop/FYP/FYP_Code/code/data_clean'
csv_file = os.path.join(folder_path,"lookup_table.csv")
combined_df.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

##############################################################################

### calculate portfolio return ###
def calculate_portfolio_returns(weights_df , returns_df):
    portfolio_returns = (weights_df * returns_df).sum(axis=1)
    return portfolio_returns
all_portfolio_returns = []
for combination in combined_df['lookup']:
    weight = portfolio_weight_obtained[combination]
    portfolio_returns = calculate_portfolio_returns(weight, test_return)
    portfolio_returns = list(portfolio_returns)
    all_portfolio_returns.append(portfolio_returns) 
combined_df['portfolio_return'] = all_portfolio_returns

### calculate base statistics ###
all_mean_return = []
all_sd_return = []
all_sharpe_ratio = []
for index, combination in enumerate(combined_df['lookup']):
    portfolio_return = combined_df['portfolio_return'][index]
    mean_return = np.mean(portfolio_return)
    sd_return = np.std(portfolio_return)
    sharpe_ratio = mean_return / sd_return
    
    all_mean_return.append(mean_return)
    all_sd_return.append(sd_return)
    all_sharpe_ratio.append(sharpe_ratio)
combined_df['mean_return'] = all_mean_return
combined_df['sd_return'] = all_sd_return
combined_df['sharpe_ratio'] = all_sharpe_ratio

### calculate max drawdown ###
def calculate_max_drawdown(monthly_return):

    cumulative_returns = (1 + monthly_return).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    return -max_drawdown
all_maxdd = []
for index, combination in enumerate(combined_df['lookup']):
    portfolio_return = combined_df['portfolio_return'][index]
    portfolio_return = pd.Series(portfolio_return)
    max_dd = calculate_max_drawdown(portfolio_return)
    all_maxdd.append(max_dd) 
combined_df['max_dd'] = all_maxdd

### calculate transaction cost ###
def calculate_transaction_costs(weights_df):

    weight_changes = weights_df.diff().abs()
    total_transaction_cost = weight_changes.sum().sum()

    return total_transaction_cost
all_tc = []
for combination in combined_df['lookup']:
    weight = portfolio_weight_obtained[combination]   
    tc = calculate_transaction_costs(weight)
    all_tc.append(tc)
combined_df['tc'] = all_tc

### calculate final investment amounts ###
monthly_returns = list(combined_df['portfolio_return'])
df = pd.DataFrame(monthly_returns).T
initial_investment = 1000
investment_trajectories = (1 + df).cumprod() * initial_investment
final_investment_amount = pd.DataFrame({'Final Investment Amount': investment_trajectories.iloc[-1]})
combined_df['final_investment_amount'] = final_investment_amount

# function to calculate percentiles
def calculate_percentiles(lst):
    return np.percentile(lst, 5), np.percentile(lst, 95)
# calculate 5th and 95th percentile returns
combined_df[['5th_percentile', '95th_percentile']] = combined_df['portfolio_return'].apply(
    lambda x: pd.Series(calculate_percentiles(x))
)

folder_path = 'data_clean'
csv_file = os.path.join(folder_path,"final_compiled_results.csv")
combined_df.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

 ##############################################################################

# generate pivot table for presentation --- sharpe ratio #
result = combined_df.pivot_table(index=['Distribution', 'Risk'], 
                                 columns='Method', 
                                 values='sharpe_ratio').reset_index()

result['diff'] = result['mpt'] - result['proposed']
result.reset_index(drop = True, inplace = True)
print(result)

# generate pivot table for presentation --- max drawdown #
result = combined_df.pivot_table(index=['Distribution', 'Risk'], 
                                 columns='Method', 
                                 values='max_dd').reset_index()

result['diff'] = result['mpt'] - result['proposed']
result.reset_index(drop = True, inplace = True)
print(result)

# generate pivot table for presentation --- transaction cost #
result = combined_df.pivot_table(index=['Distribution', 'Risk'], 
                                 columns='Method', 
                                 values='tc').reset_index()

result['diff'] = result['mpt'] - result['proposed']
result.reset_index(drop = True, inplace = True)
print(result)

##############################################################################

# filter_columns = ['Method','Distribution','Risk','sharpe_ratio']
# combined_df[filter_columns]
# combined_df.sort_values(by = 'sharpe_ratio', ascending = False)

##############################################################################

### Individual Boxplots ###
import matplotlib.pyplot as plt
import seaborn as sns

monthly_returns = list(combined_df['portfolio_return'])
df = pd.DataFrame(monthly_returns).T

plt.figure(figsize=(14, 8))
sns.boxplot(data=df, palette="Set3")

plt.title('Box Plots of Monthly Returns for 24 Series', fontsize=16)
plt.xlabel('Series', fontsize=14)
plt.ylabel('Monthly Returns', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels if needed

plt.tight_layout()
plt.show()

##############################################################################

### Grouped Boxplot of Returns --- Method ###

group_labels = combined_df['Method'].values
assert len(group_labels) == df.shape[1]

groups_df = pd.DataFrame({'Group': group_labels})
df_grouped = pd.concat([groups_df, df], axis=1)
df_melted = df_grouped.melt(id_vars='Group', var_name='Series', value_name='Monthly Return')

plt.figure(figsize=(14, 8))
sns.violinplot(x='Group', y='Monthly Return', data=df_melted, palette="Set3")

plt.title('Grouped Violin Plots of Monthly Returns --- Method', fontsize=16)
plt.xlabel('Group', fontsize=14)
plt.ylabel('Monthly Returns', fontsize=14)

plt.tight_layout()
plt.savefig('Grouped_Violinplot_Returns_Method.png', dpi=300, bbox_inches='tight')
plt.show()


### Grouped Boxplot of Returns --- Distribution ###

group_labels = combined_df['Distribution'].values
assert len(group_labels) == df.shape[1]

groups_df = pd.DataFrame({'Group': group_labels})
df_grouped = pd.concat([groups_df, df], axis=1)
df_melted = df_grouped.melt(id_vars='Group', var_name='Series', value_name='Monthly Return')

plt.figure(figsize=(14, 8))
sns.violinplot(x='Group', y='Monthly Return', data=df_melted, palette="Set3")

plt.title('Grouped Violin Plots of Monthly Returns --- Distribution', fontsize=16)
plt.xlabel('Group', fontsize=14)
plt.ylabel('Monthly Returns', fontsize=14)

plt.tight_layout()
plt.savefig('Grouped_Violinplot_Returns_Distribution.png', dpi=300, bbox_inches='tight')
plt.show()


### Grouped Boxplot of Returns --- Risk ###

group_labels = combined_df['Risk'].values
assert len(group_labels) == df.shape[1]

groups_df = pd.DataFrame({'Group': group_labels})
df_grouped = pd.concat([groups_df, df], axis=1)
df_melted = df_grouped.melt(id_vars='Group', var_name='Series', value_name='Monthly Return')

plt.figure(figsize=(14, 8))
sns.violinplot(x='Group', y='Monthly Return', data=df_melted, palette="Set3")

plt.title('Grouped Violin Plots of Monthly Returns --- Risk', fontsize=16)
plt.xlabel('Group', fontsize=14)
plt.ylabel('Monthly Returns', fontsize=14)

plt.tight_layout()
plt.savefig('Grouped_Violinplot_Returns_Risk.png', dpi=300, bbox_inches='tight')
plt.show()

##############################################################################


### Investment Trajectories ###
initial_investment = 1000
investment_trajectories = (1 + df).cumprod() * initial_investment
print(investment_trajectories)

plt.figure(figsize=(12, 6))

investment_trajectories.columns = combined_df['lookup']
for column in investment_trajectories.columns:
    plt.plot(investment_trajectories.index, investment_trajectories[column], label=column)

plt.title('Investment Trajectories')
plt.xlabel('Time Period')
plt.ylabel('Investment Amount ($)')
plt.legend(title='Investment Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside the plot
plt.grid()

plt.tight_layout()  # Adjust layout to make room for the legend
plt.savefig('Investment_Trajectories.png', dpi=300, bbox_inches='tight')
plt.show()


### Final Investment Amounts ###
final_investment_amount = pd.DataFrame({'Final Investment Amount': investment_trajectories.iloc[-1]})
final_investment_amount['Method'] = combined_df['Method'].values
final_investment_amount['Distribution'] = combined_df['Distribution'].values
final_investment_amount['Risk'] = combined_df['Risk'].values
final_investment_amount = final_investment_amount.reset_index(drop=True)
final_investment_amount = final_investment_amount[['Method','Distribution','Risk','Final Investment Amount']]
final_investment_amount = final_investment_amount.sort_values(by = 'Final Investment Amount', ascending = False)
print(final_investment_amount)