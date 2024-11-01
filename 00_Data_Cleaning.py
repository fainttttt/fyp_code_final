import os
import yfinance as yf
import pandas as pd
from scipy.interpolate import CubicSpline

# Set working directory
os.chdir('/Users/fabian/Desktop/FYP/FYP_Code/code')
print("Current Working Directory:", os.getcwd())

# Define time frame
start_time = '1998-12-22'
end_time = '2023-12-31'

# S&P Sectors tickers
sector_tickers = {
    'Technology': ['XLK'],
    'Healthcare': ['XLV'],
    'Financials': ['XLF'],
    'Energy': ['XLE'],
    'Materials': ['XLB'],
    'ConsumerDiscretionary': ['XLY'], 
    'Industrials': ['XLI'],
    'Utilities': ['XLU'],
    'ConsumerStaples': ['XLP']
}

# Download sector data from Yahoo Finance - Closing Price
sector_df = pd.DataFrame()
for sector, tickers in sector_tickers.items():
    for ticker in tickers:
        data = yf.download(ticker, start=start_time, end=end_time)['Close']
        sector_df[f"{sector}_{ticker}"] = data

# Prepare sector DataFrame for monthly returns
sector_df['Date'] = sector_df.index
sector_df = sector_df.reset_index(drop=True)
sector_df['Date'] = pd.to_datetime(sector_df['Date'])
sector_df['YearMonth'] = sector_df['Date'].dt.to_period('M')

# Extract first date of each month
first_dates_index = sector_df.groupby('YearMonth')['Date'].idxmin()
monthly_sector_df = sector_df.loc[first_dates_index].reset_index(drop=True)

# Drop unnecessary columns and calculate monthly returns
monthly_sector_df.set_index("Date", inplace=True)
monthly_sector_df.drop(columns=['YearMonth'], inplace = True)
monthly_sector_return = monthly_sector_df.pct_change()
monthly_sector_return = monthly_sector_return.iloc[1:]
monthly_sector_return['year'] = monthly_sector_return.index.year
monthly_sector_return['month'] = monthly_sector_return.index.month
monthly_sector_return.reset_index(drop = True, inplace=True)

# Save sector monthly returns to CSV
monthly_sector_return.to_csv('data_clean/9_Sectors_Ticker_Monthly_Returns.csv', index=False)

# Load and clean Fama-French data
# ff_rates = pd.read_csv('data_raw/F-F_Research_Data_Factors.CSV', skiprows=3).iloc[0:1175, :]
# ff_rates['year'] = pd.to_numeric(ff_rates.iloc[:, 0]) // 100
# ff_rates['month'] = pd.to_numeric(ff_rates.iloc[:, 0]) % 100
# ff_rates = ff_rates.drop(columns=[ff_rates.columns[0]])

# Load and clean CPI data
cpi_data = pd.read_csv('data_raw/US_CPI.csv')
cpi_data['Date'] = pd.to_datetime(cpi_data['DATE'])
cpi_data['year'] = cpi_data['Date'].dt.year
cpi_data['month'] = cpi_data['Date'].dt.month
cpi_data = cpi_data.rename(columns={'CPIAUCNS': 'CPI'}).drop(columns=['DATE'])
cpi_data.drop(columns=['Date'], inplace = True)

# Load and clean unemployment data
unemployment_data = pd.read_csv('data_raw/US_UNEMPLOYMENT.csv')
unemployment_data['Date'] = pd.to_datetime(unemployment_data['DATE'])
unemployment_data['year'] = unemployment_data['Date'].dt.year
unemployment_data['month'] = unemployment_data['Date'].dt.month
unemployment_data = unemployment_data.rename(columns={'UNRATENSA': 'Unemployment_Rate'}).drop(columns=['DATE'])
unemployment_data.drop(columns=['Date'], inplace = True)

# Load and clean GDP data, interpolate to monthly
gdp_data = pd.read_csv('data_raw/US_GDP.csv')
gdp_data['DATE'] = pd.to_datetime(gdp_data['DATE'])
gdp_data['year'] = gdp_data['DATE'].dt.year
gdp_data['month'] = gdp_data['DATE'].dt.month
gdp_data.rename(columns={'NA000334Q':'GDP'}, inplace=True)
gdp_data['quarter'] = pd.to_datetime(gdp_data[['year','month']].assign(day=1))
quarterly_dates_ordinal = gdp_data['quarter'].map(pd.Timestamp.toordinal)
cs = CubicSpline(quarterly_dates_ordinal, gdp_data['GDP'])
monthly_dates = pd.date_range(start=gdp_data['quarter'].iloc[0], end=gdp_data['quarter'].iloc[-1], freq='MS')
monthly_dates_ordinal = monthly_dates.map(pd.Timestamp.toordinal)
monthly_gdp = cs(monthly_dates_ordinal)
interpolated_gdp_df = pd.DataFrame({'Date': monthly_dates, 'Interpolated GDP': monthly_gdp})
interpolated_gdp_df['Date'] = pd.to_datetime(interpolated_gdp_df['Date'])
interpolated_gdp_df['year'] = interpolated_gdp_df['Date'].dt.year
interpolated_gdp_df['month'] = interpolated_gdp_df['Date'].dt.month
interpolated_gdp_df.drop(columns=['Date'], inplace = True)

# Merge all datasets (FF, CPI, Unemployment, GDP)
merged_df = cpi_data.merge(unemployment_data, on=['year', 'month'], how='inner') \
                    .merge(interpolated_gdp_df, on=['year', 'month'], how='inner')

# Add growth and rate changes, filter for years >= 1999
merged_df['CPI growth'] = merged_df['CPI'].pct_change()
merged_df['Change in Unemployment Rate'] = merged_df['Unemployment_Rate'].diff()
merged_df['GDP growth'] = merged_df['Interpolated GDP'].pct_change()
merged_df = merged_df[merged_df['year'] >= 1999].reset_index(drop=True)
merged_df = merged_df[['CPI growth','Change in Unemployment Rate','GDP growth','year','month']]

# Save merged economic indicators to CSV
merged_df.to_csv('data_clean/economic_indicators.csv', index=False)