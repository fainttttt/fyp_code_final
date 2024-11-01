import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

alpha = 0.05
x = np.linspace(-3, 3, 1000)
pdf = norm.pdf(x, loc=0, scale=1)


### graph on optimizing left tail ###
percentile_alpha = norm.ppf(alpha, loc=0, scale=1)
plt.plot(x, pdf, color='black', label='Density Function')

# Shade the area beyond the alpha percentile (left or right tail based on alpha)
if alpha <= 0.5:
    plt.fill_between(x, pdf, where=(x <= percentile_alpha), color='red', alpha=0.5, 
                     label=f'Risk Zone ($\\alpha$={alpha})')
else:
    plt.fill_between(x, pdf, where=(x >= percentile_alpha), color='red', alpha=0.5, 
                     label=f'Risk Zone ($\\alpha$={alpha})')

# Add a vertical line at the alpha percentile
plt.axvline(percentile_alpha, color='darkblue', linestyle='dashed', linewidth=2, 
            label=f'{int(alpha * 100)}th Percentile Cutoff')

# Add annotation to explain the significance of the tail risk and cutoff
if alpha <= 0.5:
    plt.text(percentile_alpha - 1.5, 0.15, f'Left Tail ($\\alpha$={alpha})\nExtreme Losses',
             color='darkred', fontsize=10, bbox=dict(facecolor='white', edgecolor='darkred', boxstyle='round,pad=0.5'))
else:
    plt.text(percentile_alpha - 0.5, 0.15, f'Right Tail ($\\alpha$={alpha})\nExtreme Gains',
             color='darkred', fontsize=10, bbox=dict(facecolor='white', edgecolor='darkred', boxstyle='round,pad=0.5'))

plt.xlabel('Returns')
plt.ylabel('Probability of Returns')
plt.title(f'Distribution of Portfolio Returns')
plt.legend()
plt.savefig('proposed_algorithm_left_tail.png', dpi=300, bbox_inches='tight')
plt.show()


### graph on optimizing right tail ###
percentile_alpha = norm.ppf(1 - alpha, loc=0, scale=1)
plt.plot(x, pdf, color='black', label='Density Function')

# Shade the area beyond the (1 - alpha) percentile (right tail)
plt.fill_between(x, pdf, where=(x >= percentile_alpha), color='green', alpha=0.5, 
                 label=f'Reward Zone ($\\alpha$={alpha})')

# Add a vertical line at the (1 - alpha) percentile
plt.axvline(percentile_alpha, color='darkblue', linestyle='dashed', linewidth=2, 
            label=f'{int((1 - alpha) * 100)}th Percentile Cutoff')

# Add annotation to explain the significance of the tail risk and cutoff
plt.text(percentile_alpha + 0.5, 0.15, f'Right Tail ($\\alpha$={alpha})\nExtreme Gains',
         color='darkgreen', fontsize=10, bbox=dict(facecolor='white', edgecolor='darkgreen', boxstyle='round,pad=0.5'))

plt.xlabel('Returns')
plt.ylabel('Probability of Returns')
plt.title(f'Distribution of Portfolio Returns')
plt.legend()
plt.savefig('proposed_algorithm_right_tail.png', dpi=300, bbox_inches='tight')
plt.show()


### MPT - efficient frontier ###
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Define returns and risk (standard deviation) for two assets
returns = np.array([0.08, 0.12])  # Expected returns for Asset 1 and Asset 2
risks = np.array([0.1, 0.2])       # Standard deviations for Asset 1 and Asset 2

# Define the risk-free rate
risk_free_rate = 0.03

# Generate portfolios
num_portfolios = 100
results = []

for w in np.linspace(0, 1, num_portfolios):
    # Portfolio return and risk for the given weight of Asset 1
    portfolio_return = w * returns[0] + (1 - w) * returns[1]
    portfolio_risk = np.sqrt((w ** 2 * risks[0] ** 2) + ((1 - w) ** 2 * risks[1] ** 2))
    results.append((portfolio_risk, portfolio_return))

# Convert results to a numpy array for easier manipulation
results = np.array(results)

# Extract portfolio risks and returns
portfolio_risks = results[:, 0]
portfolio_returns = results[:, 1]

# Find the maximum Sharpe ratio point
sharpe_ratios = (portfolio_returns - risk_free_rate) / portfolio_risks
max_sharpe_idx = np.argmax(sharpe_ratios)
max_sharpe_return = portfolio_returns[max_sharpe_idx]
max_sharpe_risk = portfolio_risks[max_sharpe_idx]
s_r = sharpe_ratios[max_sharpe_idx]

# Calculate the slope of the tangent line (Sharpe Ratio)
tangent_slope = s_r

# Create the plot
plt.figure(figsize=(12, 8))
plt.scatter(portfolio_risks, portfolio_returns, color='lightblue', marker='o', s=100, label='Portfolios')

# Plot the efficient frontier
plt.plot(portfolio_risks, portfolio_returns, color='green', lw=2, label='Efficient Frontier')

# Plot the maximum Sharpe ratio point
plt.scatter(max_sharpe_risk, max_sharpe_return, color='red', marker='*', s=300, label='Maximum Sharpe Ratio', edgecolor='black')

# Label the maximum Sharpe ratio point
plt.text(max_sharpe_risk + 0.01, max_sharpe_return, 'Max Sharpe Ratio', fontsize=12, color='red', weight='bold')

# Plot 100% allocation points for each asset
plt.scatter(risks[0], returns[0], color='orange', s=200, label='100% Asset 1', edgecolor='black')
plt.scatter(risks[1], returns[1], color='purple', s=200, label='100% Asset 2', edgecolor='black')

# Draw the tangent line from the risk-free rate
x_tangent = np.linspace(0, max_sharpe_risk + 0.15, 100)
y_tangent = risk_free_rate + tangent_slope * (x_tangent - 0)  # Pass through the risk-free rate at y-axis

plt.plot(x_tangent, y_tangent, color='blue', linestyle='dashed', label='Tangential Line (Sharpe Ratio)')

# Add the risk-free rate line
plt.axhline(y=risk_free_rate, color='purple', linestyle='dotted', label='Risk-Free Rate')

# Add labels and title
plt.title('Efficient Frontier', fontsize=18, weight='bold')
plt.xlabel('Portfolio Risk (Standard Deviation)', fontsize=14)
plt.ylabel('Portfolio Return', fontsize=14)
plt.legend(loc='upper left', fontsize=12)
plt.xlim(0, max_sharpe_risk + 0.15)  # Extend x-axis for clarity
plt.ylim(risk_free_rate, max_sharpe_return + 0.05)  # Extend y-axis for clarity

# Display the plot
plt.tight_layout()
plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
plt.show()