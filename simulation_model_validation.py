import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/Users/fabian/Desktop/FYP/FYP_Code/code')
current_dir = os.getcwd()

relative_file_path = os.path.join(current_dir, 'data_clean', 'simulation_model_validation.csv')
model_validation_df = pd.read_csv(relative_file_path)

# Step 1: Create intervals
num_intervals = 10
interval_size = len(model_validation_df) // num_intervals

# Step 2: Calculate average RMSE for each model within the intervals
averaged_rmse = model_validation_df.groupby(np.arange(len(model_validation_df)) // interval_size).mean()

# Step 3: Plot the average RMSE for each model across the intervals
plt.figure(figsize=(12, 6))
plt.plot(averaged_rmse.index, averaged_rmse['rmse_norm'], label='RMSE Norm', marker='o')
plt.plot(averaged_rmse.index, averaged_rmse['rmse_t'], label='RMSE t', marker='o')
plt.plot(averaged_rmse.index, averaged_rmse['rmse_sn'], label='RMSE SN', marker='o')
plt.title('Average RMSE Comparison of Simulation Models Across Intervals')
plt.xlabel('Intervals')
plt.ylabel('Average RMSE')
plt.xticks(ticks=averaged_rmse.index, labels=[f'{i+1}' for i in averaged_rmse.index])
plt.legend()
plt.grid()
plt.savefig('RMSE_Simulation_Models.png', dpi=300, bbox_inches='tight')
plt.show()