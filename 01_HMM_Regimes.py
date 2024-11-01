import os
os.chdir('/Users/fabian/Desktop/FYP/FYP_Code/code')
print("Current Working Directory:", os.getcwd())

import numpy as np
import pandas as pd
from hmmlearn import hmm

current_dir = os.getcwd()

relative_file_path = os.path.join(current_dir, 'data_clean', '9_Sectors_Ticker_Monthly_Returns.csv')
monthly_return = pd.read_csv(relative_file_path)

relative_file_path = os.path.join(current_dir, 'data_clean', 'economic_indicators.csv')
economic_indicators = pd.read_csv(relative_file_path)

indicators = economic_indicators.iloc[:300,:3]
indicators['CPI growth'] = indicators['CPI growth'] * 100 # convert to percentage unit to make it comparable with unemployment rates
indicators['GDP growth'] = indicators['GDP growth'] * 100 # convert to percentage unit to make it comparable with unemployment rates
indicators.head()

# how to determine which states are bull and bear respectively
# lower cpi growth = lower inflation = bull
# lower unemployment rate = bull
# higher gdp growth = bull
# if two or more criteria is met, consider that state as bull

def check_bull_bear(state0_mean, state1_mean):
    check = (state0_mean < state1_mean).astype(int)
    check[2] = 1 - check[2] # flip third element
    if np.sum(check) >= 2:
        return "bull" # sum >= 2 means state0 is bull
    else:
        return "bear"
    
n_states = 2 # bull-bear regimes
window_size = 36

model = hmm.GaussianHMM(n_states,"diag",1000)

hmm_result = []

# tuple = (chosen_state, transition matrix, state means, state covariances)

for start in range(0,len(indicators) - window_size): # 0 to 263
    
    np.random.seed(start)
    train_data = indicators.iloc[start:start+window_size].values
    test_data = indicators.iloc[start+window_size].values.reshape(1,-1)
    model.fit(train_data)
    
    state0_mean = model.means_[0,:]
    state1_mean = model.means_[1,:]
    
    if model.predict(test_data) == 0:
        chosen_state = check_bull_bear(state0_mean,state1_mean)
    else:
        chosen_state = check_bull_bear(state1_mean,state0_mean)
    
    item_to_append = (chosen_state,model.transmat_,model.means_,model.covars_)
    hmm_result.append(item_to_append)
    
    print(f"Completed iteration {start}")

# pre-determined parameter
# risk_averse = 1
# risk_seeking = -1

risk_aversion_parameter = []

for state, transition_matrix, state_mean, state_covariance in hmm_result:
    if state == "bear":
        # risk_aversion_parameter.append(risk_averse) # risk averse
        risk_aversion_parameter.append('bear') # risk averse
    else:
        # risk_aversion_parameter.append(risk_seeking) # risk seeking
        risk_aversion_parameter.append('bull') # risk averse

risk_aversion_parameter_df = pd.DataFrame(risk_aversion_parameter, columns = ["HMM_regimes"])
test_date = monthly_return.iloc[36:,9:]
test_date.reset_index(drop=True,inplace=True)
risk_aversion_parameter_with_date = pd.concat([risk_aversion_parameter_df,test_date],axis=1)

folder_path = 'data_clean'
csv_file = os.path.join(folder_path,"01_HMM_Regimes_BullBear.csv")
risk_aversion_parameter_with_date.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")