import pandas as pd
from pathlib import Path
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
# Set up directories
home_dir = Path.home()
work_dir = home_dir / 'consumer_sentiment'
data_dir = work_dir / 'data' / 'raw'
output_dir = work_dir / 'output'
gas_figures = output_dir / 'umich_figures' / 'gaspx'

# Load data
umich_microdata_df = pd.read_csv(data_dir / 'umich_microdata.csv')
usa_retail_all_df = pd.read_csv(data_dir / 'usa_all_retail.csv')

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

# drop the missing/don't know codes:
values_to_drop = [-997, 996, 998, 999]
umich_microdata_df = (umich_microdata_df[~umich_microdata_df[['GAS1', 'GAS5']]
                                         .isin(values_to_drop).any(axis=1)])

# now let's do a basic plot for each: 
for gas_var, horizon_label in zip(['GAS1', 'GAS5'], ['1-Year', '5-Year']):
    umich_microdata_df[gas_var] = pd.to_numeric(umich_microdata_df[gas_var], errors='coerce') * 0.01
    collapsed_df = (umich_microdata_df.groupby(['date', 'dem', 'rep', 'independent'])
                    [[gas_var]].mean().reset_index())
    plt.figure(figsize=(10, 6))
    colors = plt.get_cmap('tab10')
    for i, (party, party_label) in enumerate(zip(['dem', 'rep', 'independent'], 
            ['Democrats', 'Republicans', 'Independents'])):
        party_df = collapsed_df[collapsed_df[party] == 1]
        color = colors(i)
        plt.plot(party_df['date'], party_df[gas_var], label=party_label, color=color, 
            linewidth=0.25)

        party_df[f'{gas_var}_3ma'] = party_df[gas_var].rolling(window=3).mean()
        plt.plot(party_data['date'], party_df[f'{gas_var}_3ma'], 
                label=f'{party_label} 3-Month MA', color=color, linestyle='-', linewidth=1.5)

    plt.axvline(pd.to_datetime('2021-01-01'), color='black', linewidth=2.5,
                linestyle='--', label='Joe Biden Inauguration')
    plt.axhline(0, color='black', linewidth=2.5, linestyle='-')

    plt.grid(True)
    plt.title(f'Expected {horizon_label} Change in Gas Prices by Partisan Affiliation')
    plt.xlabel('Date')
    plt.ylabel(f'Expected {horizon_label} Change in Gas Prices')
    plt.legend()
    plt.savefig(f'{gas_figures}/{gas_var.lower()}_party_comp.png')
    plt.show()

# convert date to datetime (so that we get a clean pd.merge later)
usa_retail_all_df['date'] = pd.to_datetime(usa_retail_all_df['date'])
for var, period in zip(['gas 1-month pct. change', 'gas 1-yr pct. change', 
                        'gas 5-yr pct. change'], [1, 12, 60]):
    usa_retail_all_df[var] = (pd.to_numeric(usa_retail_all_df['usa retail all']
                                            .pct_change(periods=period))) * 100
# merge two datasets together
umich_microdata_df = pd.merge(umich_microdata_df, usa_retail_all_df, on='date', how='outer')
# convert all columns to numeric
umich_microdata_df = umich_microdata_df.apply(pd.to_numeric, errors='coerce')
umich_microdata_df['date'] = pd.to_datetime(umich_microdata_df['date'])
# cut off dataset at 2017-10-01, which is when UMich started asking about
# partisan affiliation
start_date = pd.to_datetime('2017-10-01')
umich_filtered_df = umich_microdata_df[umich_microdata_df['date']>=start_date]
# now let's define the Biden presidency through a dummy variable
biden_start = pd.to_datetime('2021-01-01')
umich_filtered_df['biden'] = (((umich_filtered_df['date'] >= biden_start))
                              .astype(int))
# create 2 temporary datasets corresponding to the biden presidency and pre-biden presidency
biden_df = umich_filtered_df[umich_filtered_df['biden'] == 1]
pre_biden_df = umich_filtered_df[umich_filtered_df['biden'] == 0]

# define function to calculate z-score
def calculate_zscore(series):
    return (series - series.mean()) / series.std()

# let's plot z-scores over time: 
def plot_gas_chg_time_series(df, zscore_vars, gaspx_chg_vars, title, period, savepath):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    for var in zscore_vars:
        ax1.plot(df['date'], df[var], linestyle='-', label=f'{var} Z-Score')
    ax1.axvline(pd.to_datetime('2021-01-01'), color='black', linewidth=2.5, 
        linestyle='--', label='Joe Biden Inauguration')
    ax1.axhline(0, color='black', linewidth=2.5, linestyle='-')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Standard Deviations from Mean % Change')
    ax1.tick_params(axis='y')
    ax1.grid(True)
    ax2 = ax1.twinx()
    for var in gaspx_chg_vars:
        ax2.plot(df['date'], df[var], linestyle='--', label={var}, color='purple', 
                 linewidth=1.5)
    ax2.set_ylabel(f'% {period.title()} change')
    ax2.tick_params(axis='y', colors='black')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{gas_figures}/{savepath}.png')
    plt.show()

periods = ['1-month', '1-yr', '5-yr']

for period in periods:
    umich_filtered_df.loc[umich_filtered_df['biden'] == 1, 
                        f'{period} gas pct. change zscore, biden'] = (
                        calculate_zscore(biden_df[f'gas {period} pct. change']))
    umich_filtered_df.loc[umich_filtered_df['biden'] == 0, 
                        f'{period} gas pct. change zscore, pre-biden'] = (
                        calculate_zscore(pre_biden_df[f'gas {period} pct. change']))
    zscore_vars = [f'{period} gas pct. change zscore, biden', 
    f'{period} gas pct. change zscore, pre-biden']
    gaspx_chg_vars = [f'gas {period} pct. change']
    plot_gas_chg_time_series(umich_filtered_df, zscore_vars, gaspx_chg_vars, 
                             f'{period} Gas Price Percent Change Z-Scores', period,
                             f'{period}_gas_zscore_time_series')

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
    for gas_var, horizon in zip(['GAS1', 'GAS5'], ['1', '5']):
        for biden_value, biden_desc in zip([1, 0], ['biden', 'pre-biden']):
            df_biden = df[df['biden'] == biden_value]
            for model_label, regressor in zip(['model_month_chg', 'model_precise_chg', 'model_level'],
                                            [f'1-month gas pct. change zscore, {biden_desc}', 
                                            f'{horizon}-yr gas pct. change zscore, {biden_desc}',
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
                    'horizon': f'{horizon}-yr.',
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

models=['model_month_chg', 'model_precise_chg', 'model_level']
model_labs=['Gas Price Monthly Difference Model', 'Gas Price Precise Difference Model',
           'Gas Price Levels Model']
gas_vars=['GAS1', 'GAS5']
horizons = ['1', '5']
gas_labels=['1-Year Gas Price Expected Change', '5-Year Gas Price Expected Change']
parties=['dem', 'rep', 'independent']
party_labels=['Democrats', 'Republicans', 'Independents']

for gas_var, gas_label, horizon in zip(gas_vars, gas_labels, horizons):
    for model, model_lab in zip(models, model_labs):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (party, party_label) in enumerate(zip(parties, party_labels)):
            party_df = coefficients_df[(coefficients_df['model'] == model) 
                                       & (coefficients_df['party'] == party)]
        
            biden_1_val = (party_df[party_df['biden'] == 1][party_df['horizon']==f'{horizon}-yr.']
                        ['beta_1'].values[0])
            biden_0_val = (party_df[party_df['biden'] == 0]['beta_1']
                        [party_df['horizon']==f'{horizon}-yr.'].values[0])
        
            biden_1_se = (party_df[party_df['biden'] == 1][party_df['horizon']==f'{horizon}-yr.']
                        ['beta_1_se'].values[0])
            biden_0_se = (party_df[party_df['biden'] == 0][party_df['horizon']==f'{horizon}-yr.']
                        ['beta_1_se'].values[0])
        
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
