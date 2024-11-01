from model_mpt import *
from model_proposed import *

os.chdir('/Users/fabian/Desktop/FYP/FYP_Code/code')
print("Current Working Directory:", os.getcwd())

current_dir = os.getcwd()

relative_file_path = os.path.join(current_dir, 'data_clean', '9_Sectors_Ticker_Monthly_Returns.csv')
monthly_return = pd.read_csv(relative_file_path)

relative_file_path = os.path.join(current_dir, 'data_clean', '01_HMM_Regimes_BullBear.csv')
hmm_regimes = pd.read_csv(relative_file_path)

all_historical_data = monthly_return.iloc[:,:9]
num_simulations = 500
num_samples = 100
sample_size = 20
simulation_method = 'multivariate_normal'
df = 5
window_size = 36
risk_aversion_method = 'hidden_markov_model' # or 'most_recent_performance'

# additional parameter for mpt model
risk_averse = 1
risk_seeking = -1 # may not be possible in mpt algorithm
hmm_regime_parameter = np.where(hmm_regimes.iloc[:, 0] == 'bull', risk_seeking, risk_averse)
hmm_regimes.iloc[:,0] = hmm_regime_parameter

# additional parameter for proposed model
alpha = 0.05

# saving output    
folder_path = 'portfolio_weights'

##### simulation 1: normal distribution #####

### model mpt ###
optimal_weights_mpt_normal_mrp, utility_results_mpt_normal_mrp = rolling_portfolio_optimization_mpt(monthly_return_entire_data = all_historical_data,
                                                                                                    num_simulations = num_simulations,
                                                                                                    num_samples = num_samples,
                                                                                                    sample_size = sample_size,
                                                                                                    simulation_method = 'multivariate_normal',
                                                                                                    df = 5,
                                                                                                    window_size = window_size,
                                                                                                    risk_aversion_method = 'most_recent_performance')

optimal_weights_mpt_normal_hmm, utility_results_mpt_normal_hmm = rolling_portfolio_optimization_mpt(monthly_return_entire_data = all_historical_data,
                                                                                                    num_simulations = num_simulations,
                                                                                                    num_samples = num_samples,
                                                                                                    sample_size = sample_size,
                                                                                                    simulation_method = 'multivariate_normal',
                                                                                                    df = 5,
                                                                                                    window_size = window_size,
                                                                                                    risk_aversion_method = 'hidden_markov_model')

optimal_weights_mpt_normal_risk_seeking, utility_results_mpt_normal_risk_seeking = rolling_portfolio_optimization_mpt(monthly_return_entire_data = all_historical_data,
                                                                                                                      num_simulations = num_simulations,
                                                                                                                      num_samples = num_samples,
                                                                                                                      sample_size = sample_size,
                                                                                                                      simulation_method = 'multivariate_normal',
                                                                                                                      df = 5,
                                                                                                                      window_size = window_size,
                                                                                                                      risk_aversion_method = 'all_risk_seeking')

optimal_weights_mpt_normal_risk_averse, utility_results_mpt_normal_risk_averse = rolling_portfolio_optimization_mpt(monthly_return_entire_data = all_historical_data,
                                                                                                                    num_simulations = num_simulations,
                                                                                                                    num_samples = num_samples,
                                                                                                                    sample_size = sample_size,
                                                                                                                    simulation_method = 'multivariate_normal',
                                                                                                                    df = 5,
                                                                                                                    window_size = window_size,
                                                                                                                    risk_aversion_method = 'all_risk_averse')

### model proposed ###
all_result_proposed_normal_mrp, risk_attitudes_proposed_normal_mrp = rolling_portfolio_optimization_proposed(monthly_return_entire_data = all_historical_data,
                                                                                                             num_simulations = num_simulations,
                                                                                                             num_samples = num_samples,
                                                                                                             sample_size = sample_size,
                                                                                                             simulation_method = 'multivariate_normal',
                                                                                                             df = 5,
                                                                                                             window_size = window_size,
                                                                                                             alpha = alpha,
                                                                                                             risk_aversion_method = 'most_recent_performance')

all_result_proposed_normal_hmm, risk_attitudes_proposed_normal_hmm = rolling_portfolio_optimization_proposed(monthly_return_entire_data = all_historical_data,
                                                                                                             num_simulations = num_simulations,
                                                                                                             num_samples = num_samples,
                                                                                                             sample_size = sample_size,
                                                                                                             simulation_method = 'multivariate_normal',
                                                                                                             df = 5,
                                                                                                             window_size = window_size,
                                                                                                             alpha = alpha,
                                                                                                             risk_aversion_method = 'hidden_markov_model')

all_result_proposed_normal_risk_seeking, risk_attitudes_proposed_normal_risk_seeking = rolling_portfolio_optimization_proposed(monthly_return_entire_data = all_historical_data,
                                                                                                                               num_simulations = num_simulations,
                                                                                                                               num_samples = num_samples,
                                                                                                                               sample_size = sample_size,
                                                                                                                               simulation_method = 'multivariate_normal',
                                                                                                                               df = 5,
                                                                                                                               window_size = window_size,
                                                                                                                               alpha = alpha,
                                                                                                                               risk_aversion_method = 'all_risk_seeking')

all_result_proposed_normal_risk_averse, risk_attitudes_proposed_normal_risk_averse = rolling_portfolio_optimization_proposed(monthly_return_entire_data = all_historical_data,
                                                                                                                            num_simulations = num_simulations,
                                                                                                                            num_samples = num_samples,
                                                                                                                            sample_size = sample_size,
                                                                                                                            simulation_method = 'multivariate_normal',
                                                                                                                            df = 5,
                                                                                                                            window_size = window_size,
                                                                                                                            alpha = alpha,
                                                                                                                            risk_aversion_method = 'all_risk_averse')

# unpacking output
column_name = np.array(monthly_return.iloc[:,:9].columns)

mrp = [item[0] for item in all_result_proposed_normal_mrp]
optimal_weights_proposed_normal_mrp = pd.DataFrame(columns = column_name)
for count in range(len(mrp)):
    optimal_weights_proposed_normal_mrp.loc[count+1] = mrp[count]
    
hmm = [item[0] for item in all_result_proposed_normal_hmm]
optimal_weights_proposed_normal_hmm = pd.DataFrame(columns = column_name)
for count in range(len(hmm)):
    optimal_weights_proposed_normal_hmm.loc[count] = hmm[count]
    
risk_seeking = [item[0] for item in all_result_proposed_normal_risk_seeking]
optimal_weights_proposed_normal_risk_seeking = pd.DataFrame(columns = column_name)
for count in range(len(risk_seeking)):
    optimal_weights_proposed_normal_risk_seeking.loc[count] = risk_seeking[count]
    
risk_averse = [item[0] for item in all_result_proposed_normal_risk_averse]
optimal_weights_proposed_normal_risk_averse = pd.DataFrame(columns = column_name)
for count in range(len(risk_averse)):
    optimal_weights_proposed_normal_risk_averse.loc[count] = risk_averse[count]

equal_weight = pd.DataFrame(1/9, index=range(264), columns = column_name)
    
csv_file = os.path.join(folder_path,"mpt_normal_mrp.csv")
optimal_weights_mpt_normal_mrp.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"mpt_normal_hmm.csv")
optimal_weights_mpt_normal_hmm.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"mpt_normal_risk_seeking.csv")
optimal_weights_mpt_normal_risk_seeking.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"mpt_normal_risk_averse.csv")
optimal_weights_mpt_normal_risk_averse.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"proposed_normal_mrp.csv")
optimal_weights_proposed_normal_mrp.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"proposed_normal_hmm.csv")
optimal_weights_proposed_normal_hmm.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"proposed_normal_risk_seeking.csv")
optimal_weights_proposed_normal_risk_seeking.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"proposed_normal_risk_averse.csv")
optimal_weights_proposed_normal_risk_averse.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"equal_weight.csv")
equal_weight.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")


##### simulation 2: t distribution #####

### model mpt ###
optimal_weights_mpt_t_mrp, utility_results_mpt_t_mrp = rolling_portfolio_optimization_mpt(monthly_return_entire_data = all_historical_data,
                                                                                          num_simulations = num_simulations,
                                                                                          num_samples = num_samples,
                                                                                          sample_size = sample_size,
                                                                                          simulation_method = 'multivariate_t',
                                                                                          df = 5,
                                                                                          window_size = window_size,
                                                                                          risk_aversion_method = 'most_recent_performance')

optimal_weights_mpt_t_hmm, utility_results_mpt_t_hmm = rolling_portfolio_optimization_mpt(monthly_return_entire_data = all_historical_data,
                                                                                          num_simulations = num_simulations,
                                                                                          num_samples = num_samples,
                                                                                          sample_size = sample_size,
                                                                                          simulation_method = 'multivariate_t',
                                                                                          df = 5,
                                                                                          window_size = window_size,
                                                                                          risk_aversion_method = 'hidden_markov_model')

optimal_weights_mpt_t_risk_seeking, utility_results_mpt_t_risk_seeking = rolling_portfolio_optimization_mpt(monthly_return_entire_data = all_historical_data,
                                                                                                            num_simulations = num_simulations,
                                                                                                            num_samples = num_samples,
                                                                                                            sample_size = sample_size,
                                                                                                            simulation_method = 'multivariate_t',
                                                                                                            df = 5,
                                                                                                            window_size = window_size,
                                                                                                            risk_aversion_method = 'all_risk_seeking')

optimal_weights_mpt_t_risk_averse, utility_results_mpt_t_risk_averse = rolling_portfolio_optimization_mpt(monthly_return_entire_data = all_historical_data,
                                                                                                          num_simulations = num_simulations,
                                                                                                          num_samples = num_samples,
                                                                                                          sample_size = sample_size,
                                                                                                          simulation_method = 'multivariate_t',
                                                                                                          df = 5,
                                                                                                          window_size = window_size,
                                                                                                          risk_aversion_method = 'all_risk_averse')

### model proposed ###
all_result_proposed_t_mrp, risk_attitudes_proposed_t_mrp = rolling_portfolio_optimization_proposed(monthly_return_entire_data = all_historical_data,
                                                                                                   num_simulations = num_simulations,
                                                                                                   num_samples = num_samples,
                                                                                                   sample_size = sample_size,
                                                                                                   simulation_method = 'multivariate_t',
                                                                                                   df = 5,
                                                                                                   window_size = window_size,
                                                                                                   alpha = alpha,
                                                                                                   risk_aversion_method = 'most_recent_performance')

all_result_proposed_t_hmm, risk_attitudes_proposed_t_hmm = rolling_portfolio_optimization_proposed(monthly_return_entire_data = all_historical_data,
                                                                                                   num_simulations = num_simulations,
                                                                                                   num_samples = num_samples,
                                                                                                   sample_size = sample_size,
                                                                                                   simulation_method = 'multivariate_t',
                                                                                                   df = 5,
                                                                                                   window_size = window_size,
                                                                                                   alpha = alpha,
                                                                                                   risk_aversion_method = 'hidden_markov_model')

all_result_proposed_t_risk_seeking, risk_attitudes_proposed_t_risk_seeking = rolling_portfolio_optimization_proposed(monthly_return_entire_data = all_historical_data,
                                                                                                                     num_simulations = num_simulations,
                                                                                                                     num_samples = num_samples,
                                                                                                                     sample_size = sample_size,
                                                                                                                     simulation_method = 'multivariate_t',
                                                                                                                     df = 5,
                                                                                                                     window_size = window_size,
                                                                                                                     alpha = alpha,
                                                                                                                     risk_aversion_method = 'all_risk_seeking')

all_result_proposed_t_risk_averse, risk_attitudes_proposed_t_risk_averse = rolling_portfolio_optimization_proposed(monthly_return_entire_data = all_historical_data,
                                                                                                                   num_simulations = num_simulations,
                                                                                                                   num_samples = num_samples,
                                                                                                                   sample_size = sample_size,
                                                                                                                   simulation_method = 'multivariate_t',
                                                                                                                   df = 5,
                                                                                                                   window_size = window_size,
                                                                                                                   alpha = alpha,
                                                                                                                   risk_aversion_method = 'all_risk_averse')

# unpacking output
column_name = np.array(monthly_return.iloc[:,:9].columns)

mrp_t = [item[0] for item in all_result_proposed_t_mrp]
optimal_weights_proposed_t_mrp = pd.DataFrame(columns = column_name)
for count in range(len(mrp_t)):
    optimal_weights_proposed_t_mrp.loc[count+1] = mrp_t[count]
    
hmm_t = [item[0] for item in all_result_proposed_t_hmm]
optimal_weights_proposed_t_hmm = pd.DataFrame(columns = column_name)
for count in range(len(hmm_t)):
    optimal_weights_proposed_t_hmm.loc[count] = hmm_t[count]
    
risk_seeking_t = [item[0] for item in all_result_proposed_t_risk_seeking]
optimal_weights_proposed_t_risk_seeking = pd.DataFrame(columns = column_name)
for count in range(len(risk_seeking_t)):
    optimal_weights_proposed_t_risk_seeking.loc[count] = risk_seeking_t[count]
    
risk_averse_t = [item[0] for item in all_result_proposed_t_risk_averse]
optimal_weights_proposed_t_risk_averse = pd.DataFrame(columns = column_name)
for count in range(len(risk_averse_t)):
    optimal_weights_proposed_t_risk_averse.loc[count] = risk_averse_t[count]

csv_file = os.path.join(folder_path,"mpt_t_mrp.csv")
optimal_weights_mpt_t_mrp.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"mpt_t_hmm.csv")
optimal_weights_mpt_t_hmm.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"mpt_t_risk_seeking.csv")
optimal_weights_mpt_t_risk_seeking.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"mpt_t_risk_averse.csv")
optimal_weights_mpt_t_risk_averse.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"proposed_t_mrp.csv")
optimal_weights_proposed_t_mrp.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"proposed_t_hmm.csv")
optimal_weights_proposed_t_hmm.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"proposed_t_risk_seeking.csv")
optimal_weights_proposed_t_risk_seeking.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"proposed_t_risk_averse.csv")
optimal_weights_proposed_t_risk_averse.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")


##### simulation 3: skew normal distribution #####

### model mpt ###
optimal_weights_mpt_skew_normal_mrp, utility_results_mpt_skew_normal_mrp = rolling_portfolio_optimization_mpt(monthly_return_entire_data = all_historical_data,
                                                                                                              num_simulations = num_simulations,
                                                                                                              num_samples = num_samples,
                                                                                                              sample_size = sample_size,
                                                                                                              simulation_method = 'skew_normal',
                                                                                                              df = 5,
                                                                                                              window_size = window_size,
                                                                                                              risk_aversion_method = 'most_recent_performance')

optimal_weights_mpt_skew_normal_hmm, utility_results_mpt_skew_normal_hmm = rolling_portfolio_optimization_mpt(monthly_return_entire_data = all_historical_data,
                                                                                                              num_simulations = num_simulations,
                                                                                                              num_samples = num_samples,
                                                                                                              sample_size = sample_size,
                                                                                                              simulation_method = 'skew_normal',
                                                                                                              df = 5,
                                                                                                              window_size = window_size,
                                                                                                              risk_aversion_method = 'hidden_markov_model')

optimal_weights_mpt_skew_normal_risk_seeking, utility_results_mpt_skew_normal_risk_seeking = rolling_portfolio_optimization_mpt(monthly_return_entire_data = all_historical_data,
                                                                                                                                num_simulations = num_simulations,
                                                                                                                                num_samples = num_samples,
                                                                                                                                sample_size = sample_size,
                                                                                                                                simulation_method = 'skew_normal',
                                                                                                                                df = 5,
                                                                                                                                window_size = window_size,
                                                                                                                                risk_aversion_method = 'all_risk_seeking')

optimal_weights_mpt_skew_normal_risk_averse, utility_results_mpt_skew_normal_risk_averse = rolling_portfolio_optimization_mpt(monthly_return_entire_data = all_historical_data,
                                                                                                                              num_simulations = num_simulations,
                                                                                                                              num_samples = num_samples,
                                                                                                                              sample_size = sample_size,
                                                                                                                              simulation_method = 'skew_normal',
                                                                                                                              df = 5,
                                                                                                                              window_size = window_size,
                                                                                                                              risk_aversion_method = 'all_risk_averse')

### model proposed ###
all_result_proposed_skew_normal_mrp, risk_attitudes_proposed_skew_normal_mrp = rolling_portfolio_optimization_proposed(monthly_return_entire_data = all_historical_data,
                                                                                                                       num_simulations = num_simulations,
                                                                                                                       num_samples = num_samples,
                                                                                                                       sample_size = sample_size,
                                                                                                                       simulation_method = 'skew_normal',
                                                                                                                       df = 5,
                                                                                                                       window_size = window_size,
                                                                                                                       alpha = alpha,
                                                                                                                       risk_aversion_method = 'most_recent_performance')

all_result_proposed_skew_normal_hmm, risk_attitudes_proposed_skew_normal_hmm = rolling_portfolio_optimization_proposed(monthly_return_entire_data = all_historical_data,
                                                                                                                       num_simulations = num_simulations,
                                                                                                                       num_samples = num_samples,
                                                                                                                       sample_size = sample_size,
                                                                                                                       simulation_method = 'skew_normal',
                                                                                                                       df = 5,
                                                                                                                       window_size = window_size,
                                                                                                                       alpha = alpha,
                                                                                                                       risk_aversion_method = 'hidden_markov_model')

all_result_proposed_skew_normal_risk_seeking, risk_attitudes_proposed_skew_normal_risk_seeking = rolling_portfolio_optimization_proposed(monthly_return_entire_data = all_historical_data,
                                                                                                                                         num_simulations = num_simulations,
                                                                                                                                         num_samples = num_samples,
                                                                                                                                         sample_size = sample_size,
                                                                                                                                         simulation_method = 'skew_normal',
                                                                                                                                         df = 5,
                                                                                                                                         window_size = window_size,
                                                                                                                                         alpha = alpha,
                                                                                                                                         risk_aversion_method = 'all_risk_seeking')

all_result_proposed_skew_normal_risk_averse, risk_attitudes_proposed_skew_normal_risk_averse = rolling_portfolio_optimization_proposed(monthly_return_entire_data = all_historical_data,
                                                                                                                                       num_simulations = num_simulations,
                                                                                                                                       num_samples = num_samples,
                                                                                                                                       sample_size = sample_size,
                                                                                                                                       simulation_method = 'skew_normal',
                                                                                                                                       df = 5,
                                                                                                                                       window_size = window_size,
                                                                                                                                       alpha = alpha,
                                                                                                                                       risk_aversion_method = 'all_risk_averse')

# unpacking output
column_name = np.array(monthly_return.iloc[:,:9].columns)

mrp_skew_normal = [item[0] for item in all_result_proposed_skew_normal_mrp]
optimal_weights_proposed_skew_normal_mrp = pd.DataFrame(columns = column_name)
for count in range(len(mrp_skew_normal)):
    optimal_weights_proposed_skew_normal_mrp.loc[count+1] = mrp_skew_normal[count]
    
hmm_skew_normal = [item[0] for item in all_result_proposed_skew_normal_hmm]
optimal_weights_proposed_skew_normal_hmm = pd.DataFrame(columns = column_name)
for count in range(len(hmm_skew_normal)):
    optimal_weights_proposed_skew_normal_hmm.loc[count] = hmm_skew_normal[count]
    
risk_seeking_skew_normal = [item[0] for item in all_result_proposed_skew_normal_risk_seeking]
optimal_weights_proposed_skew_normal_risk_seeking = pd.DataFrame(columns = column_name)
for count in range(len(risk_seeking_skew_normal)):
    optimal_weights_proposed_skew_normal_risk_seeking.loc[count] = risk_seeking_skew_normal[count]
    
risk_averse_skew_normal = [item[0] for item in all_result_proposed_skew_normal_risk_averse]
optimal_weights_proposed_skew_normal_risk_averse = pd.DataFrame(columns = column_name)
for count in range(len(risk_averse_skew_normal)):
    optimal_weights_proposed_skew_normal_risk_averse.loc[count] = risk_averse_skew_normal[count]

# saving output    
folder_path = 'portfolio_weights'

csv_file = os.path.join(folder_path,"mpt_skew_normal_mrp.csv")
optimal_weights_mpt_skew_normal_mrp.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"mpt_skew_normal_hmm.csv")
optimal_weights_mpt_skew_normal_hmm.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"mpt_skew_normal_risk_seeking.csv")
optimal_weights_mpt_skew_normal_risk_seeking.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"mpt_skew_normal_risk_averse.csv")
optimal_weights_mpt_skew_normal_risk_averse.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"proposed_skew_normal_mrp.csv")
optimal_weights_proposed_skew_normal_mrp.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"proposed_skew_normal_hmm.csv")
optimal_weights_proposed_skew_normal_hmm.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"proposed_skew_normal_risk_seeking.csv")
optimal_weights_proposed_skew_normal_risk_seeking.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")

csv_file = os.path.join(folder_path,"proposed_skew_normal_risk_averse.csv")
optimal_weights_proposed_skew_normal_risk_averse.to_csv(csv_file, index = False)
print(f"DataFrame saved as CSV in: {csv_file}")