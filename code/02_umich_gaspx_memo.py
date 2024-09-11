import pandas as pd
from pathlib import Path
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
# let's create a set of locals referring to our directory and working directory 
home_dir = Path.home()
work_dir = (home_dir / 'consumer_sentiment')
data = (work_dir / 'data')
raw_data = (data / 'raw')
code = Path.cwd() 
output = (work_dir / 'output')
gas_figures = (output / 'umich_figures' / 'gaspx')

umich_microdata_df = pd.read_csv(f'{raw_data}/umich_microdata.csv', header=0)

# let's convert YYYYMM into a proper datetime variable
umich_microdata_df['year'] = pd.to_numeric(umich_microdata_df['YYYY'])
umich_microdata_df['month'] = umich_microdata_df['YYYYMM'] % 100
umich_microdata_df['day'] = 1
umich_microdata_df['date'] = pd.to_datetime(umich_microdata_df[['year', 'month', 'day']])

# let's create a set of dummies representing political affiliation
umich_microdata_df['dem'] = np.where(umich_microdata_df['POLAFF'] == 2, 1, 0)
umich_microdata_df['rep'] = np.where(umich_microdata_df['POLAFF'] == 1, 1, 0)
umich_microdata_df['independent'] = np.where(umich_microdata_df['POLAFF'] == 3, 1, 0)
umich_microdata_df['other'] = np.where(umich_microdata_df['POLAFF'] == 7, 1, 0)
'''
umich_microdata_df['strong dem'] = np.where(umich_microdata_df['POLDEM'] == 1, 1, 0)
umich_microdata_df['strong rep'] = np.where(umich_microdata_df['POLREP'] == 1, 1, 0)
umich_microdata_df['gas prices go up next 5 years'] = np.where(umich_microdata_df['GASPX1'] == 1, 1, 0)
umich_microdata_df['gas prices constant next 5 years'] = np.where(umich_microdata_df['GASPX1'] == 3, 1, 0)
umich_microdata_df['gas prices go down next 5 years'] = np.where(umich_microdata_df['GASPX1'] == 5, 1, 0)
'''
# drop the missing/don't know codes:
values_to_drop = [-997, 996, 998, 999]
for var in ['GAS1', 'GAS5']:
    umich_microdata_df = umich_microdata_df[~umich_microdata_df[var].isin(values_to_drop)]

usa_retail_all_df = pd.read_csv(f'{raw_data}/usa_all_retail.csv')
usa_retail_all_df['date'] = pd.to_datetime(usa_retail_all_df['date'])
for var, period in zip(['gas price change', 'gas1 change', 'gas5 change'],
                       [1, 12, 60]):
    usa_retail_all_df[var] = pd.to_numeric(usa_retail_all_df['usa retail all'].diff(periods=period))

umich_microdata_df = pd.merge(umich_microdata_df, usa_retail_all_df, on='date', how='outer')

for var in umich_microdata_df.columns:
    umich_microdata_df[var] = pd.to_numeric(umich_microdata_df[var], errors='coerce')

umich_microdata_df['date'] = pd.to_datetime(umich_microdata_df['date'])
start_date = pd.to_datetime('2017-10-01')
umich_filtered_df = umich_microdata_df[umich_microdata_df['date']>=start_date]

biden_start = pd.to_datetime('2021-01-01')
biden_end = umich_filtered_df['date'].max()
umich_filtered_df['biden'] = ((umich_filtered_df['date'] >= biden_start) & (umich_filtered_df['date'] <= biden_end)).astype(int)
umich_filtered_df['biden'] = umich_filtered_df['biden'].fillna(0)

for var in ['GAS1', 'GAS5']:
    umich_filtered_df[var] = umich_filtered_df[var]*0.01
coefficients_gas1 = {'date': [], 'dem_coef': [], 'rep_coef': []}
coefficients_gas5 = {'date': [], 'dem_coef': [], 'rep_coef': []}

# define a function to run regressions
def run_regression(df, dependent_var, independent_vars):
    X = sm.add_constant(df[independent_vars])
    y = df[dependent_var]
    model = sm.OLS(y, X).fit()
    return model

# Loop Through Dates and Run Regressions
for date in umich_filtered_df['date'].unique():
    temp_df = umich_filtered_df[umich_filtered_df['date'] == date]
    print(temp_df)
    for var, coefficients_df_temp  in zip(['GAS1', 'GAS5'], 
                                                         [coefficients_gas1, coefficients_gas5]):
        temp_df_gas = temp_df.dropna(subset=[var])
        if not temp_df_gas.empty:
            model = run_regression(temp_df_gas, var, ['dem', 'rep', 'independent', 'other'])
            coefficients_df_temp['date'].append(date)
            coefficients_df_temp['dem_coef'].append(model.params['dem'])
            coefficients_df_temp['rep_coef'].append(model.params['rep'])

# Convert the coefficients dictionaries to DataFrames
coefficients_gas1_df = pd.DataFrame(coefficients_gas1)
coefficients_gas5_df = pd.DataFrame(coefficients_gas5)

# Plotting the coefficients over time for GAS1
coefficients_dfs = [(coefficients_gas1_df, 'GAS1', '1-Year Gas Price Expectation', 'gas1_umich.png'),
                    (coefficients_gas5_df, 'GAS5', '5-Year Gas Price Expectation', 'gas5_umich.png')]

# Loop through the dataframes and plot
for df, gas_label, title_label, filename in coefficients_dfs:
    plt.figure(figsize=(14, 7))
    plt.plot(df['date'], df['dem_coef'], label='DEM Coefficient', marker='o')
    plt.plot(df['date'], df['rep_coef'], label='REP Coefficient', marker='x')
    plt.title(f'Partisan Identification Effect on {title_label}')
    plt.xlabel('Date')
    plt.ylabel('Coefficient')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.axvline(pd.to_datetime('2021-01-01'), color='black', 
                linewidth=2.5, linestyle='--', label='Joe Biden Inauguration')
    plt.axvline(pd.to_datetime('2020-11-01'), color='black', 
            linewidth=2.5, linestyle='--', label='Joe Biden Election')
    plt.axhline(0, color='black', linewidth=2.5, linestyle='-')
    plt.legend()
    plt.savefig(f'{gas_figures}/{filename}')
    plt.show()

# now collapse on average prediction
collapsed_df = umich_filtered_df.groupby(['date', 'dem', 'rep', 'independent']).mean().reset_index()

for var, lab in zip(['GAS1', 'GAS5'], ['1-Year', '5-Year']):
    df_dem = collapsed_df[collapsed_df['dem'] == 1]
    df_rep = collapsed_df[collapsed_df['rep'] == 1]
    df_ind = collapsed_df[collapsed_df['independent'] == 1]
    plt.figure(figsize=(10, 6))
    plt.plot(df_dem['date'], df_dem[var], label='Democrat', color='blue')
    plt.plot(df_rep['date'], df_rep[var], label='Republican', color='red')
    plt.plot(df_ind['date'], df_ind[var], label='Independent', color='green')
    plt.title(f'Average {lab} Gas Price Expectations by Political Affiliation')
    plt.xlabel('Date')
    plt.ylabel(f'{lab} Gas Price Predicted Change (nominal $)')
    plt.axvline(pd.to_datetime('2021-01-01'), color='black', 
            linewidth=2.5, linestyle='--', label='Joe Biden Inauguration')
    plt.axvline(pd.to_datetime('2020-11-01'), color='black', 
            linewidth=2.5, linestyle='--', label='Joe Biden Election')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{gas_figures}/{var.lower()}_levels.png')
    plt.show()
# now let's run a separate regression for independents, democrats, 
# and republicans using the actual price of gasoline
dem_df = pd.DataFrame()
rep_df = pd.DataFrame()
ind_df = pd.DataFrame()
groups = {
    'dem': dem_df,
    'rep': rep_df,
    'independent': ind_df
}
coefficients_list = []
for group, df in groups.items():
    df = pd.concat([df, umich_filtered_df[umich_filtered_df[group] == 1]])
    for gas_var in ['GAS1', 'GAS5']:
        for biden_value in [1, 0]:
            df_biden = df[df['biden'] == biden_value]
            for model_label, regressor in zip(['model_month_chg', 'model_precise_chg', 'model_level'],
                                            ['gas price change', f'{gas_var.lower()} change',
                                            'usa retail all']):
                model = run_regression(df_biden, gas_var, regressor)
                X = sm.add_constant(df_biden[[regressor]])
                umich_filtered_df.loc[
                    (umich_filtered_df[group] == 1) & (umich_filtered_df['biden']==biden_value),
                       f'{gas_var.lower()} residual {model_label}'] = model.resid
                umich_filtered_df.loc[
                    (umich_filtered_df[group] == 1) & (umich_filtered_df['biden']==biden_value),
                       f'{gas_var.lower()} predicted value {model_label}'] = model.predict(X)
                coefficients = {
                    'party': f'{group}',
                    'biden': biden_value,
                    'model': model_label,
                    'horizon': gas_var,
                    'beta_0': model.params['const'],
                    'beta_1': model.params[regressor],
                    'beta_1_se': model.bse[regressor]
                }
                coefficients_list.append(coefficients)
coefficients_df = pd.DataFrame(coefficients_list, 
                               columns=[
                                   'model', 'horizon', 'beta_0',
                                   'beta_1', 'party', 'beta_1_se', 'biden'
                                   ])
coefficients_df.to_csv(f'{gas_figures}/coefficients_table.csv', index=False)

# now let's clean the residual/actual/predicted table from above:
cols_keep = ['date', 'CASEID', 'usa retail all', 'GAS1', 'GAS5', 'gas price change',
            'gas1 change', 'gas5 change', 'gas1 residual model_precise_chg',
            'gas1 predicted value model_month_chg', 'gas1 residual model_month_chg',
            'gas1 predicted value model_precise_chg', 'gas1 residual model_level',
            'gas1 predicted value model_level', 'gas5 residual model_month_chg',
            'gas5 predicted value model_month_chg',
            'gas5 residual model_precise_chg',
            'gas5 predicted value model_precise_chg', 'gas5 residual model_level',
            'gas5 predicted value model_level', 'biden', 'rep', 'independent', 'dem']
residual_filtered_df = umich_filtered_df[cols_keep]
residual_filtered_df = residual_filtered_df[
    (residual_filtered_df['dem'] != 0) | 
    (residual_filtered_df['rep'] != 0) | 
    (residual_filtered_df['independent'] != 0)
]
residual_filtered_df.to_csv(f'{data}/umich_filtered_residual_analysis.csv', index=False)

models=['model_month_chg', 'model_precise_chg', 'model_level']
model_labs=['Gas Price Monthly Difference Model', 'Gas Price Precise Difference Model',
           'Gas Price Levels Model']
gas_vars=['GAS1', 'GAS5']
gas_labels=['1-Year Gas Price Expected Change', '5-Year Gas Price Expected Change']
parties=['dem', 'rep', 'independent']
party_labels=['Democrats', 'Republicans', 'Independents']

for gas_var, gas_label in zip(gas_vars, gas_labels):
    for model, model_lab in zip(models, model_labs):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (party, party_label) in enumerate(zip(parties, party_labels)):
            party_df = coefficients_df[(coefficients_df['model'] == model) 
                                       & (coefficients_df['party'] == party)]
        
            biden_1_val = party_df[party_df['biden'] == 1][party_df['horizon']==gas_var]['beta_1'].values[0]
            biden_0_val = party_df[party_df['biden'] == 0]['beta_1'][party_df['horizon']==gas_var].values[0]
        
            biden_1_se = party_df[party_df['biden'] == 1][party_df['horizon']==gas_var]['beta_1_se'].values[0]
            biden_0_se = party_df[party_df['biden'] == 0][party_df['horizon']==gas_var]['beta_1_se'].values[0]
        
            x_pos_biden_0 = i - 0.2
            x_pos_biden_1 = i + 0.2
        
            ax.bar(x_pos_biden_0, biden_0_val, 0.4, yerr=1.96 * biden_0_se, 
               color='r', alpha=0.6, label=f'Trump Presidency' if i == 0 else "")
            ax.bar(x_pos_biden_1, biden_1_val, 0.4, yerr=1.96 * biden_1_se, 
               color='b', alpha=0.6, label=f'Biden Presidency' if i == 0 else "")
            
        ax.set_xticks(np.arange(len(parties)))
        ax.set_xticklabels(party_labels)
        #ax.set_xlabel('Political Party')
        ax.set_ylabel('Point Estimate')
        ax.set_title(f'{gas_label} Estimates by Partisan Affiliation for {model_lab}')
        ax.legend(loc='upper left')
        plt.grid(True)
        plt.savefig(f'{gas_figures}/{model}_{gas_var.lower()}_party_comp.png')
        plt.show()
 
plt.figure(figsize=(12, 6))
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(umich_filtered_df['date'], umich_filtered_df['usa retail all'], 
         linewidth=3, label='Gas Price Level', color='black')
ax1.set_xlabel('Date')
ax1.set_ylabel('Gas Price Level')
ax1.tick_params(axis='y')

ax1.axvline(pd.to_datetime('2021-01-01'), color='black', 
            linewidth=2.5, linestyle='--', label='Joe Biden Inauguration')
ax1.axvline(pd.to_datetime('2020-11-01'), color='black', 
            linewidth=2.5, linestyle='--', label='Joe Biden Election')
ax2 = ax1.twinx()
ax2.plot(umich_filtered_df['date'], umich_filtered_df['gas1 change'], 
         label='1-Year Gas Price Historical Change', color='green')
ax2.plot(umich_filtered_df['date'], umich_filtered_df['gas price change'], 
         label='Monthly Level Difference in Gas Prices', color='orange')
ax2.plot(umich_filtered_df['date'], umich_filtered_df['gas5 change'], 
         label='5-Year Gas Price Historical Change', color='red')
ax2.set_ylabel('Change/Difference in Gas Prices')
ax2.tick_params(axis='y')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
ax1.grid(True, axis='both', which='both')

plt.title('Comparing Gas Price Regressors Over Time')
plt.tight_layout()
plt.savefig(f'{gas_figures}/regressor_comp.png')
plt.show()

# build table of residuals (see above)
tables = {}
for model in models:
    for gas_var in gas_vars:
        table = pd.DataFrame(index=parties, columns=[1, 0]) 
        for party in parties:
            for biden_value in [1, 0]:
                df = residual_filtered_df[(residual_filtered_df['biden'] == biden_value) & 
                                               (residual_filtered_df[party] == 1)]
                average_residual = df[f'{gas_var.lower()} residual {model}'].mean()
                '''
                plt.figure(figsize=(10, 6))
                plt.hist(df[f'{gas_var.lower()} residual {model}'], 
                         bins=30, edgecolor='black', alpha=0.7)
                plt.axvline(average_residual, color='red', 
                            linewidth=2, linestyle='--', label=f'Mean Residual: {average_residual:.2f}')
                plt.title(f'Histogram of {gas_var.lower()} Residuals for {model}')
                plt.grid(True)
                plt.show()
                '''
                table.loc[party, biden_value] = average_residual
        table = table.applymap(lambda x: f'{x:.2e}')
        table_name = f'{model}_{gas_var}_table'
        tables[table_name] = table
for table_name, table in tables.items():
    print(f"Table: {table_name}")
    print(table)