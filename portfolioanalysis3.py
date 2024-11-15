import os
import pandas as pd
import numpy as np


# obtain hmm regimes
relative_file_path = os.path.join(os.getcwd(), 'data_clean', '01_HMM_Regimes_BullBear.csv')
hmm_regimes = pd.read_csv(relative_file_path)


# obtain test return
relative_file_path = os.path.join(os.getcwd(), 'data_clean', '9_Sectors_Ticker_Monthly_Returns.csv')
monthly_return = pd.read_csv(relative_file_path)
test_return = monthly_return.iloc[36:,:9]
test_return.reset_index(drop = True, inplace = True) ## filter for required monthly returns only


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


# calculate portfolio return
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



# analysis on risk attitudes - hmm

# collect indices
bull_indexes = pd.Series(hmm_regimes[hmm_regimes['HMM_regimes'] == 'bull'].index.tolist())
bear_indexes = pd.Series(hmm_regimes[hmm_regimes['HMM_regimes'] == 'bear'].index.tolist())

bull_components = []
bear_components = []
bull_exp = []
bear_exp = []
overall_exp = []

# filter for hmm portfolios
hmm_portfolio_return_df = combined_df[combined_df['Risk'] == 'hmm']
hmm_portfolio_return_df.reset_index(drop=True, inplace=True)

for index, row in hmm_portfolio_return_df.iterrows():
    portfolio_return = row['portfolio_return']
    
    bull_return = [portfolio_return[i] for i in bull_indexes]
    bear_return = [portfolio_return[i] for i in bear_indexes]
    
    bull_mean = np.mean(bull_return)
    bear_mean = np.mean(bear_return)
    overall_mean = np.mean(portfolio_return)
    
    bull_components.append(bull_return)
    bear_components.append(bear_return)
    bull_exp.append(bull_mean)
    bear_exp.append(bear_mean)
    overall_exp.append(overall_mean)
    
hmm_portfolio_return_df['bull_component'] = bull_components
hmm_portfolio_return_df['bear_component'] = bear_components
hmm_portfolio_return_df['Risk Seeking'] = bull_exp
hmm_portfolio_return_df['Risk Averse'] = bear_exp
hmm_portfolio_return_df['Overall'] = overall_exp

print(hmm_portfolio_return_df)

hmm_portfolio_return_df[['Method','Distribution','Risk Averse','Risk Seeking','Overall']].sort_values(['Method','Distribution'])

# # analysis on risk attitudes - mrp

mrp_portfolio_return_df = combined_df[combined_df['Risk'] == 'mrp']
mrp_portfolio_return_df.reset_index(drop=True, inplace=True)

RA = []
RS = []
RA_exp = []
RS_exp = []
overall_mean = []

for index, row in mrp_portfolio_return_df.iterrows():
    portfolio_return = row['portfolio_return']
    
    risk_averse_return = []
    risk_seeking_return = []
    
    if len(portfolio_return) > 0:
        if portfolio_return[0] > 0:
            risk_averse_return.append(portfolio_return[0])
        else:
            risk_seeking_return.append(portfolio_return[0])

        for i in range(len(portfolio_return)-1):
            if portfolio_return[i] > 0:
                risk_averse_return.append(portfolio_return[i+1])
            else:
                risk_seeking_return.append(portfolio_return[i+1])

        RA_exp_mean = np.mean(risk_averse_return)
        RS_exp_mean = np.mean(risk_seeking_return)
        all_mean = np.mean(portfolio_return)
        
        RA.append(risk_averse_return)
        RS.append(risk_seeking_return)
        RA_exp.append(RA_exp_mean)
        RS_exp.append(RS_exp_mean)
        overall_mean.append(all_mean)

mrp_portfolio_return_df['RS'] = RS
mrp_portfolio_return_df['RA'] = RA

mrp_portfolio_return_df['RA_exp'] = RA_exp
mrp_portfolio_return_df['RS_exp'] = RS_exp
mrp_portfolio_return_df['Overall_Mean'] = overall_mean

print(mrp_portfolio_return_df)

mrp_portfolio_return_df[['Method','Distribution','RA_exp','RS_exp','Overall_Mean']].sort_values(['Method','Distribution'])