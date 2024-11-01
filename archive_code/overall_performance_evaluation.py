import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

os.chdir('/Users/fabian/Desktop/FYP/FYP_Code/code')
current_dir = os.getcwd()

relative_file_path = os.path.join(current_dir, 'data_clean', '9_Sectors_Ticker_Monthly_Returns.csv')
monthly_return = pd.read_csv(relative_file_path)

test_return = monthly_return.iloc[36:,:9]
test_return.reset_index(drop = True, inplace = True)

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

def multiply_all(arr):
    result = 1
    for num in arr:
        result *= num
    return result

def sharpe_ratio(returns):
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe = mean_return / std_return
    return sharpe

def max_drawdown(returns):
    # Convert returns to a cumulative return series
    cumulative_returns = np.cumprod(1 + returns)  # This assumes returns are in decimal form
    peak = cumulative_returns[0]
    max_dd = 0

    # Iterate through cumulative returns to find the maximum drawdown
    for value in cumulative_returns:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak  # Drawdown as a fraction
        max_dd = max(max_dd, drawdown)  # Update max drawdown

    return max_dd

# Example usage
folder_path = '/Users/fabian/Desktop/FYP/FYP_Code/code/portfolio_weights'
csv_data = read_portfolio_weights(folder_path)

# Initialize a list to store the results
results = []
performance_return = {}
investment_amount = {}

# Loop through each portfolio in the dictionary
for portfolio_name, portfolio_weight in csv_data.items():
    
    # Validate if all weights sum to one
    # sum_of_weights = portfolio_weight.sum(axis=1)
    # if all(round(sum_of_weight, 3) == 1 for sum_of_weight in sum_of_weights):
    #     print("All weights sum to one!\n")

    # cumulative_portfolio = cumulative_performance * 1000 # assume initial investment at 1000 for visualization purpose
    # plt.plot(cumulative_portfolio, label=portfolio_name)  # Plot each array with a label
    # print(cumulative_portfolio)
    
    # Calculate performance (Assuming `test_return` is defined elsewhere)
    test_performance = portfolio_weight * test_return
    monthly_performance_raw = test_performance.sum(axis=1)
    monthly_performance = monthly_performance_raw + 1
    cumulative_performance = np.array(monthly_performance.cumprod())
    cumulative_investment = cumulative_performance * 1000 # assume initial investment at 1000
    
    # Calculate Performance Metrics
    overall_performance = multiply_all(monthly_performance)  # Calculate final performance
    sr = sharpe_ratio(monthly_performance_raw)  # Calculate Sharpe ratio
    max_dd = max_drawdown(monthly_performance_raw)  # Calculate max drawdown
    transaction_costs = portfolio_weight.diff().abs().sum(axis=1).sum()  # Calculate transaction cost
    
    # Append the calculated statistics to the results list
    results.append({
        'Portfolio Name': portfolio_name,
        'Sharpe Ratio': sr,
        'Mean': np.mean(monthly_performance_raw),
        'SD': np.std(monthly_performance_raw),
        'Portfolio Max Drawdown': max_dd,
        'Transaction Cost': transaction_costs
    })
    
    # Store monthly performance and investment amounts in separate dictionaries
    performance_return[portfolio_name] = np.array(monthly_performance).tolist()
    investment_amount[portfolio_name] = cumulative_investment.tolist()
    
# Convert the results list to a DataFrame
portfolio_statistics_df = pd.DataFrame(results) # performance metrics
performance_df = pd.DataFrame.from_dict(performance_return, orient='index').transpose() # visualization - monthly returns
investment_df = pd.DataFrame.from_dict(investment_amount, orient='index').transpose() # visualization - investment trajectories

# Define the orders for filtering
method_order = ['mpt', 'proposed']
distribution_order = ['normal', 'skew_normal', 't']
risk_order = ['hmm', 'mrp', 'risk_averse', 'risk_seeking']

# Create a DataFrame to store the calculated percentiles
results_df = pd.DataFrame(columns=['Method', 'Distribution', 'Risk Order', '5th Percentile (%)', '50th Percentile (%)', '95th Percentile (%)'])

# Create a figure for each distribution - monthly return
for distribution in distribution_order:
    # Create a 2x2 subplot for the current distribution
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))  # Adjusted figsize to 10x8 for better fit
    fig.suptitle(f'Distribution used for data simulation: {distribution}', fontsize=16)  # Main title for the grid

    # Create a rectangle for the outline
    outline = Rectangle((0, 0), 1, 1, transform=fig.transFigure, color='black', fill=False, linewidth=2)
    fig.patches.append(outline)  # Add the rectangle to the figure

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i, risk in enumerate(risk_order):
        # Construct column names for the two methods
        column_name_mpt = f"mpt_{distribution}_{risk}"
        column_name_proposed = f"proposed_{distribution}_{risk}"

        # Check if the columns exist
        if column_name_mpt in performance_df.columns and column_name_proposed in performance_df.columns:
            # Create a DataFrame to hold the data for the current risk order
            plot_data = pd.DataFrame({
                'Method': ['mpt'] * len(performance_df[column_name_mpt]) + ['proposed'] * len(performance_df[column_name_proposed]),
                'Values': pd.concat([performance_df[column_name_mpt], performance_df[column_name_proposed]]).values
            })

            # Create the violin plot
            sns.violinplot(data=plot_data, x='Method', y='Values', ax=axes[i], inner=None, palette=["lightblue", "lightcoral"], alpha=0.7)

            # Overlay the boxplot
            sns.boxplot(data=plot_data, x='Method', y='Values', ax=axes[i], color='black', width=0.2, showfliers=False)

            # Overlay the data points with jitter
            sns.stripplot(data=plot_data, x='Method', y='Values', ax=axes[i], color='gray', alpha=0.5, jitter=True)

            # Set titles and labels
            axes[i].set_title(f'Risk Preference Used: {risk}', fontsize=14)  # Title for each subplot
            axes[i].set_ylabel('Values', fontsize=12)  # Adjusted fontsize
            axes[i].set_xlabel('Methods', fontsize=12)  # Adjusted fontsize

            # Calculate and store key percentiles in percentage format
            for method in ['mpt', 'proposed']:
                method_data = plot_data[plot_data['Method'] == method]['Values']
                p05_val = (method_data.quantile(0.05) - 1) * 100
                p50_val = (method_data.quantile(0.50) - 1) * 100
                p95_val = (method_data.quantile(0.95) - 1) * 100

                # Round the percentile values to 3 decimal points
                p05_val = round(p05_val, 3)
                p50_val = round(p50_val, 3)
                p95_val = round(p95_val, 3)

                # Create a new DataFrame for the current row and concatenate
                new_row = pd.DataFrame({
                    'Method': [method],
                    'Distribution': [distribution],  # Add the distribution to the new row
                    'Risk Order': [risk],
                    '5th Percentile (%)': [p05_val],
                    '50th Percentile (%)': [p50_val],
                    '95th Percentile (%)': [p95_val]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)

        else:
            axes[i].set_title(f'Risk Preference Used: {risk} - Data Missing', fontsize=14)  # Adjusted title for missing data
            axes[i].set_ylabel('Values', fontsize=12)  # Adjusted fontsize
            axes[i].set_xlabel('Methods', fontsize=12)  # Adjusted fontsize
            axes[i].text(0.5, 0.5, 'No Data Available', horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)

    # Adjust layout with padding for the outer layer
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)  # Adjust the margins as needed
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to fit the title
    
    # Save the figure as a PNG file
    plt.savefig(f"{distribution}_violin_boxplots.png", format='png', dpi=300)  # Save as PNG with 300 DPI
    plt.close(fig)  # Close the figure to free up memory

# Create a figure for each distribution - investment trajectory
for distribution in distribution_order:
    # Create a 2x2 subplot for each distribution
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    fig.suptitle(f'Investment Trajectories - Distribution: {distribution}', fontsize=16)
    
    # Create a rectangle for the outline
    outline = Rectangle((0, 0), 1, 1, transform=fig.transFigure, color='black', fill=False, linewidth=2)
    fig.patches.append(outline)
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()
    
    # Iterate through risks
    for i, risk in enumerate(risk_order):
        # Generate column names for the two methods
        column_name_mpt = f"mpt_{distribution}_{risk}"
        column_name_proposed = f"proposed_{distribution}_{risk}"
        
        # Check if the columns exist in the dataframe
        if column_name_mpt in investment_df.columns and column_name_proposed in investment_df.columns:
            # Extract investment trajectories for both methods
            plot_data = pd.DataFrame({
                'Time': list(range(len(investment_df))),
                'MPT': investment_df[column_name_mpt],
                'Proposed': investment_df[column_name_proposed]
            })
            
            # Plot investment trajectories for both methods
            axes[i].plot(plot_data['Time'], plot_data['MPT'], label='MPT', color='lightblue', linewidth=2)
            axes[i].plot(plot_data['Time'], plot_data['Proposed'], label='Proposed', color='lightcoral', linewidth=2, linestyle='--')
            
            # Set titles, labels, and legends
            axes[i].set_title(f'Risk: {risk}', fontsize=14)
            axes[i].set_xlabel('Time Periods', fontsize=12)
            axes[i].set_ylabel('Investment Amount', fontsize=12)
            axes[i].legend(loc='upper left', fontsize=10)
        else:
            # If data is missing, display a message
            axes[i].set_title(f'Risk: {risk} - Data Missing', fontsize=14)
            axes[i].text(0.5, 0.5, 'No Data Available', horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)

    # Adjust layout and add padding
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save each distribution's figure
    plt.savefig(f"{distribution}_investment_trajectories.png", format='png', dpi=300)
    plt.close(fig)

# Initialize a list to store the results
final_investments = []

# Calculate final investment amounts
for method in method_order:
    for distribution in distribution_order:
        for risk in risk_order:
            # Construct the portfolio name (column name in the dataframe)
            column_name = f"{method}_{distribution}_{risk}"
            
            # Check if the column exists in the dataframe
            if column_name in investment_df.columns:
                # Extract the final investment amount (last row of the column)
                final_investment = investment_df[column_name].iloc[-1]
                
                # Store the result in a dictionary format
                final_investments.append({
                    'Portfolio': column_name,
                    'Final Investment Amount': final_investment
                })

# Convert the list of dictionaries into a dataframe
final_investment_df = pd.DataFrame(final_investments)

# Display the final investment amounts dataframe
print(final_investment_df)


### performance metric ###
# Define the order for each category
method_order = ['mpt', 'proposed']
distribution_order = ['normal', 'skew_normal', 't']
risk_order = ['hmm', 'mrp', 'risk_averse', 'risk_seeking']

# Split the 'Portfolio Name' into Method, Distribution, and Risk Preference
portfolio_statistics_df['Method'] = portfolio_statistics_df['Portfolio Name'].str.extract(r'^(.*?)_')[0]
portfolio_statistics_df['Distribution'] = portfolio_statistics_df['Portfolio Name'].str.extract(r'_(.*?)_')[0]
portfolio_statistics_df['Risk Preference'] = portfolio_statistics_df['Portfolio Name'].str.split('_', n=2).str[2]  # Get everything after the second underscore

# Map the order to numeric values
portfolio_statistics_df['Method Order'] = portfolio_statistics_df['Method'].map({method: index for index, method in enumerate(method_order)})
portfolio_statistics_df['Distribution Order'] = portfolio_statistics_df['Distribution'].map({dist: index for index, dist in enumerate(distribution_order)})
portfolio_statistics_df['Risk Order'] = portfolio_statistics_df['Risk Preference'].map({risk: index for index, risk in enumerate(risk_order)})

sorted_df = portfolio_statistics_df.sort_values(by=['Method Order', 'Distribution Order', 'Risk Order']) # sort DataFrame based on the order columns
sorted_df = sorted_df.drop(columns=['Method Order', 'Distribution Order', 'Risk Order']) # drop redundant columns

# Display the sorted DataFrame
print(sorted_df)

# Create a pivot table for Sharpe Ratio
sharpe_table = sorted_df.pivot_table(
    index=['Distribution', 'Risk Preference'], 
    columns='Method', 
    values='Sharpe Ratio'
)

# Create a pivot table for Mean Return
mean_return_table = sorted_df.pivot_table(
    index=['Distribution', 'Risk Preference'], 
    columns='Method', 
    values='Mean'
)

# Create a pivot table for SD Return
sd_return_table = sorted_df.pivot_table(
    index=['Distribution', 'Risk Preference'], 
    columns='Method', 
    values='SD'
)

# Create a pivot table for Portfolio Max Drawdown
drawdown_table = sorted_df.pivot_table(
    index=['Distribution', 'Risk Preference'], 
    columns='Method', 
    values='Portfolio Max Drawdown'
)

# Create a pivot table for Transaction Cost
cost_table = sorted_df.pivot_table(
    index=['Distribution', 'Risk Preference'], 
    columns='Method', 
    values='Transaction Cost'
)

# Calculate the differences between mpt and proposed
sharpe_diff = sharpe_table['mpt'] - sharpe_table['proposed']
drawdown_diff = drawdown_table['mpt'] - drawdown_table['proposed']
cost_diff = cost_table['mpt'] - cost_table['proposed']

# Combine differences into DataFrames
sharpe_table['Difference'] = sharpe_diff
drawdown_table['Difference'] = drawdown_diff
cost_table['Difference'] = cost_diff

# Create a new DataFrame for displaying repeated indices
sharpe_table_display = sharpe_table.reset_index()
drawdown_table_display = drawdown_table.reset_index()
cost_table_display = cost_table.reset_index()

# Set the multi-index again to show repeated first index
sharpe_table_display.set_index(['Distribution', 'Risk Preference'], inplace=True)
drawdown_table_display.set_index(['Distribution', 'Risk Preference'], inplace=True)
cost_table_display.set_index(['Distribution', 'Risk Preference'], inplace=True)

# Repeat the first index for display
sharpe_table_display.index.names = ['Distribution', 'Risk Preference']
drawdown_table_display.index.names = ['Distribution', 'Risk Preference']
cost_table_display.index.names = ['Distribution', 'Risk Preference']

# Display the tables
print("Sharpe Ratio Table:")
print(sharpe_table_display)

print("\nPortfolio Max Drawdown Table:")
print(drawdown_table)

print("\nTransaction Cost Table:")
print(cost_table)

print("\nMean Return Table:")
print(mean_return_table)

print("\nSD Return Table:")
print(sd_return_table)

sharpe_mpt = sharpe_table['mpt'].values
sharpe_proposed = sharpe_table['proposed'].values
(sharpe_mpt / sharpe_proposed)

sd_mpt = sd_return_table['mpt'].values
sd_proposed = sd_return_table['proposed'].values
(sd_mpt / sd_proposed)

mean_mpt = mean_return_table['mpt'].values
mean_proposed = mean_return_table['proposed'].values
(mean_mpt / mean_proposed)

(sharpe_mpt / sharpe_proposed)
(mean_mpt / mean_proposed) / (sd_mpt / sd_proposed)


sorted_df.head() ### sharpe ratio, max drawdown, transaction cost
final_investment_df.head() ### final investment amount
results_df.head() ### percentile

merged_df_1 = pd.merge(sorted_df, final_investment_df, left_on='Portfolio Name', right_on='Portfolio', how='inner')
merged_df_1['Distribution'] = merged_df_1['Distribution'].replace('skew', 'skew_normal')
merged_df_1['Risk Preference'] = merged_df_1['Risk Preference'].replace('normal_risk_averse', 'risk_averse')
merged_df_1['Risk Preference'] = merged_df_1['Risk Preference'].replace('normal_risk_seeking', 'risk_seeking')
merged_df_1['Risk Preference'] = merged_df_1['Risk Preference'].replace('normal_hmm', 'hmm')
merged_df_1['Risk Preference'] = merged_df_1['Risk Preference'].replace('normal_mrp', 'mrp')

final_merged_df = pd.merge(merged_df_1, results_df, 
                           left_on=['Method', 'Distribution', 'Risk Preference'], 
                           right_on=['Method', 'Distribution', 'Risk Order'], 
                           how='inner')

final_merged_df.head()

# csv_file = os.path.join(folder_path,"table_of_results.csv")
# final_merged_df.to_csv(csv_file, index = False)
# print(f"DataFrame saved as CSV in: {csv_file}")