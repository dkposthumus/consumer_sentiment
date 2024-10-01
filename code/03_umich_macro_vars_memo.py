from fredapi import Fred
import pandas as pd
from pathlib import Path
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tabulate import tabulate
import yfinance as yf
from datetime import datetime, timedelta
# let's create a set of locals referring to our directory and working directory 
home_dir = Path.home()
work_dir = (home_dir / 'consumer_sentiment')
data = (work_dir / 'data')
raw_data = (data / 'raw')
code = Path.cwd() 
output = (work_dir / 'output')
macro_figures = (output / 'umich_figures' / 'macro_vars')

fred = Fred(api_key='8905b2f5faefd705486e644f09bb8088')
# use fred function to pull data
def load_fred_series(series_id, series_name, freq, freq_name):
    series = fred.get_series(series_id).pct_change(periods=freq) * 100
    df = (series.reset_index()
          .rename(columns={'index': 'date', 0: f'{series_name}, %{freq_name}'}))
    df['date'] = pd.to_datetime(df['date'])
    return df[['date', f"{series_name}, %{freq_name}"]]
def process_sp500(start_date, end_date):
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)[['Close']].resample('M').last()
    sp500['sp500, %monthly'] = sp500['Close'].pct_change()
    sp500['sp500, %yoy'] = sp500['Close'].pct_change(12)
    sp500 = sp500.reset_index()
    sp500['date'] = sp500['Date'] - pd.offsets.MonthBegin(1)
    sp500.drop(columns=['Close', 'Date'], inplace=True)  
    return sp500
series_list = [
    ('USAUCSFRCONDOSMSAMID', 'zillow index', 12, 'yoy'), 
    ('USAUCSFRCONDOSMSAMID', 'zillow index',  1, 'monthly'),
    ('CUSR0000SETA02', 'used car cpi', 12, 'yoy'), 
    ('CUSR0000SETA02', 'used car cpi', 1, 'monthly'),
    ('A229RX0', 'real disposable income', 12, 'yoy'), 
    ('A229RX0', 'real disposable income', 1, 'monthly'),
    ('CPIAUCSL', 'headline cpi', 12, 'yoy'), 
    ('CPIAUCSL', 'headline cpi', 1, 'monthly')
]

dfs = {f"{series_name}, %{freq_name}": load_fred_series(sid, series_name, freq, freq_name) 
        for sid, series_name, freq, freq_name in series_list}

# Merge all macro variables
macro_var_df = process_sp500("2000-01-01", datetime.now().strftime('%Y-%m-%d'))
for key in dfs:
    macro_var_df = pd.merge(macro_var_df, dfs[key], on='date', how='outer')

def biden_var_creation(df):
    biden_start = pd.to_datetime('2021-01-01')
    biden_end = df['date'].max()
    df['biden'] = ((df['date'] >= biden_start) & (df['date'] <= biden_end)).astype(int)
    df['biden'] = df['biden'].fillna(0)
    return df

start_date = pd.to_datetime('2017-10-01')
macro_var_df = macro_var_df[macro_var_df['date']>=start_date]
biden_var_creation(macro_var_df)

def calculate_zscore(series):
    return (series - series.mean()) / series.std()

biden_df = macro_var_df[macro_var_df['biden'] == 1]
pre_biden_df = macro_var_df[macro_var_df['biden'] == 0]

for var, savepath in zip(['sp500', 'zillow index', 'used car cpi', 
                'real disposable income', 'headline cpi'], ['sp500', 'zillow', 'used_car', 
                'rinc', 'headline_cpi']):
    for horizon in ['monthly', 'yoy']:
        macro_var_df.loc[macro_var_df['biden'] == 1, 
                        f'{var} {horizon} pct. change zscore, biden'] = (
                        calculate_zscore(biden_df[f'{var}, %{horizon}']))
        macro_var_df.loc[macro_var_df['biden'] == 0, 
                        f'{var} {horizon} pct. change zscore, pre-biden'] = (
                        calculate_zscore(pre_biden_df[f'{var}, %{horizon}']))
        zscore_vars = [f'{var} {horizon} pct. change zscore, biden', 
                   f'{var} {horizon} pct. change zscore, pre-biden']
        chg_vars = [f'{var}, %{horizon}']
        fig, ax1 = plt.subplots(figsize=(10, 6))
        for zscore_var in zscore_vars:
            ax1.plot(macro_var_df['date'], macro_var_df[zscore_var], label=zscore_var, 
                     linewidth=1, alpha=0.6)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Z-Score')
        ax1.axhline(0, color='black', linewidth=1, linestyle='-')
        ax1.axvline(pd.to_datetime('2021-01-01'), color='black', linewidth=3, linestyle='--', 
                    label='Joe Biden Inauguration')
        ax1.grid(True)
        ax1.legend(loc='upper left')
        ax2 = ax1.twinx()
        for chg_var in chg_vars:
            ax2.plot(macro_var_df['date'], macro_var_df[chg_var], label=chg_var, 
                    linewidth=1.5, alpha=0.6, linestyle='--', color='red')
        ax2.set_ylabel('% Change')
        
        y1_min, y1_max = ax1.get_ylim()  # Get limits of the primary axis (z-scores)
        y2_min, y2_max = ax2.get_ylim()  # Get limits of the secondary axis (% changes)

        primary_zero_pos = abs(y1_min) / (y1_max - y1_min)

        secondary_range = y2_max - y2_min
        new_y2_min = -primary_zero_pos * secondary_range * 1.25  # Scale the min for the secondary axis
        new_y2_max = (1 - primary_zero_pos) * secondary_range * 1.25 # Scale the max for the secondary axis

        ax2.set_ylim(new_y2_min, new_y2_max)
        
        ax2.legend(loc='upper right')
        plt.title(f'Z-Score and Raw % Change Variables for {var.capitalize()} ({horizon.capitalize()})')
        fig.tight_layout()
        if horizon == 'monthly':
            plt.savefig(f'{macro_figures}/{savepath}_zscore.png')
            plt.show()
        if horizon == 'yoy' and var == 'headline cpi':
            
            new_y2_min = new_y2_min * 1.75
            new_y2_max = new_y2_max * 1.75

            ax2.set_ylim(new_y2_min, new_y2_max)

            plt.savefig(f'{macro_figures}/{savepath}_zscore.png')
            plt.show()
        plt.close()

# okay, macro variables are cleaned. let's import and clean the umich microdata now:
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

# so basically i need to craft the _r variable using individual observations: 
    # those are calculated as (the percentage who say variable_x is good) 
    # - (percentage who say variable_y is bad)
def plot_time_series(data, parties, party_labels, variable, title, ylabel, save, save_path):
    plt.figure(figsize=(14, 7))
    colors = plt.get_cmap('tab10')
    for i, (party, party_label) in enumerate(zip(parties, party_labels)):
        party_data = data[data[party] == 1].copy()
        party_data[f'{variable}_6ma'] = party_data[variable].rolling(window=6).mean()
        color = colors(i)
        plt.plot(party_data['date'], party_data[f'{variable}_6ma'], 
                 color=color, linewidth=2, label=f'{party_label} 6-Month MA')
        plt.plot(party_data['date'], party_data[variable], label=f'{party_label}', 
                 color=color, linewidth=0.5, alpha=0.6)
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.axvline(pd.to_datetime('2020-03-01'), color='grey', linewidth=3, 
                linestyle='--', label='COVID-19 Pandemic Outbreak')
    plt.axvline(pd.to_datetime('2020-12-01'), color='brown', linewidth=3, 
                linestyle='--', label='Joe Biden Election')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    if save==True:
        plt.savefig(save_path)
    plt.show()

panel_data_frames = []
for var, result_r, graph_label in zip(['GOVT', 'CAR', 'HOM', 'RINC', 'BAGO', 'PAGO'], 
                                      ['govt_r', 'veh_r', 'hom_r', 'rinc_r', 'bago_r', 'pago_r'],
                                      ['Government Economic Policy', 'Vehicle Buying Conditions',
                                       'Home Buying Conditions', 'Real Family Income',
                                       'Business Conditions', 'Personal Financial Situation',]):
    panel_data = []
    for date, group in umich_microdata_df.groupby('date'):
        for group_name in ['dem', 'rep', 'independent']:
            group_members = group[group[group_name] == 1]
            total_count = group_members.shape[0] 
            if total_count == 0:
                continue 
            count_1 = group_members[group_members[var] == 1].shape[0]
            percentage_1 = (count_1 / total_count) * 100
            count_5 = group_members[group_members[var] == 5].shape[0]
            percentage_5 = (count_5 / total_count) * 100
            result = (percentage_1) - (percentage_5) + 100
            panel_data.append({
                'date': date,
                group_name: 1,
                result_r: result
            })
    temp_df = pd.DataFrame(panel_data)
    plot_time_series(temp_df, ['dem', 'rep', 'independent'], ['Democrat', 'Republican', 'Independent'], 
                     result_r, f'{graph_label} Net Attitudes, by Partisan Affiliation',
                     'Good - Bad (%)', save=True, 
                     save_path=f'{macro_figures}/{result_r}_time_series.png')
    panel_data_frames.append(temp_df)
# now there are 2 trickier ones to analyze over time: 
    # PSTK (percentage chance investment will increase in next year)
average_pstk_df = pd.DataFrame()
for party in ['dem', 'rep', 'independent']:
    party_data = umich_microdata_df[umich_microdata_df[party] == 1]
    party_data = party_data[~party_data['PSTK'].isin([998, 999])]
    average_pstk = party_data.groupby('date')['PSTK'].mean().reset_index()
    average_pstk[party] = 1
    average_pstk_df = pd.concat([average_pstk_df, average_pstk])
panel_data_frames.append(average_pstk_df)
plot_time_series(average_pstk_df, ['dem', 'rep', 'independent'], 
                 ['Democrat', 'Republican', 'Independent'],
                 'PSTK', 
                 'Mean Predicted Likelihood of Positive Stock Performance, by Partisan Affiliation',
                 'Percentage', save=True, save_path=f'{macro_figures}/PSTK_time_series.png')

# now find percentage of people who answered 'NEWS1' or 'NEWS2' with 72
newsrn_u_pri_df = pd.DataFrame()
umich_microdata_df['news_72'] = (umich_microdata_df['NEWS1'] == 72) | (umich_microdata_df['NEWS2'] == 72)
parties = ['dem', 'rep', 'independent']
percentage_data =[]
for party in parties:
    grouped = umich_microdata_df.groupby(['date', party])['news_72'].mean().reset_index()
    grouped['newsrn_u_pri'] = grouped['news_72'] * 100
    grouped = grouped[grouped[party] == 1]
    grouped[party] = 1 
    percentage_data.append(grouped[['date', party, 'newsrn_u_pri']])
percentage_df = pd.concat(percentage_data)
plt.figure(figsize=(10, 6))
for party in parties:
    party_data = percentage_df[percentage_df[party] == 1]
    newsrn_u_pri_df = pd.concat([newsrn_u_pri_df, party_data])
panel_data_frames.append(newsrn_u_pri_df)
plot_time_series(newsrn_u_pri_df, ['dem', 'rep', 'independent'],
                 ['Democrat', 'Republican', 'Independent'],
                 'newsrn_u_pri', 
                 '% Reported Unfavorable News Coverage of Inflation, by Partisan Affiliation',
                 'Percentage', save=True, save_path=f'{macro_figures}/newsrn_u_pri_time_series.png')
panel_data_frames = [df.reset_index(drop=True) for df in panel_data_frames]
panel_df = pd.concat(panel_data_frames, axis=1)
panel_df = panel_df.loc[:,~panel_df.columns.duplicated()]

# now we need to run regressions: 
panel_umich_df = pd.merge(panel_df, macro_var_df, on='date', how='left')

def run_regression(df, dependent_var, independent_vars):
    X = sm.add_constant(df[independent_vars])
    y = pd.to_numeric(df[dependent_var], errors='coerce')
    X = X.dropna()
    y = y.loc[X.index]
    if X.empty or y.empty:
        return None
    model = sm.OLS(y, X).fit()
    return model
def run_and_store_regression(df, dependent_var, independent_var, 
                             party, biden_value, model_name, coefficients_list):
    model = run_regression(df, dependent_var, independent_var)
    sample_size = len(df.dropna(subset=[dependent_var, independent_var]))
    coefficients_list.append({
        'party': party,
        'biden': biden_value,
        'model': model_name,
        'umich_var': dependent_var,
        'beta_0': model.params['const'],
        'beta_1': model.params[independent_var],
        'beta_1_se': model.bse[independent_var],
        'sample size': sample_size
    })
# let's write our dictionaries:
umich_vars = ['veh_r', 'hom_r', 'rinc_r', 'PSTK', 'newsrn_u_pri']
macro_vars = ['used car cpi', 'zillow index', 'real disposable income', 'sp500', 'headline cpi']
coefficients_list = []
for outcome, regressor in zip(umich_vars, macro_vars):
    for party in ['dem', 'rep', 'independent']:
        for horizon, model_name in zip(['monthly', 'yoy'], ['monthly model', 'yoy model']):
            for biden_value, biden_descr in zip([1, 0], ['biden', 'pre-biden']):
                df_biden = (panel_umich_df[(panel_umich_df[party] == 1) 
                                              & (panel_umich_df['biden'] == biden_value)])
                run_and_store_regression(df_biden, outcome, 
                            f"{regressor} {horizon} pct. change zscore, {biden_descr}", 
                            party, biden_value, model_name, coefficients_list)
coefficients_df = pd.DataFrame(coefficients_list, 
                               columns=[
                                   'model', 'umich_var', 'beta_0',
                                   'beta_1', 'party', 'beta_1_se', 'biden', 'sample size'
                                   ])
coefficients_df.to_csv(f'{macro_figures}/coefficients_table.csv', index=False)

def plot_regression_coefficients(coefficients_df, umich_var, 
                                 umich_lab, model, model_lab, parties, party_labels, output_file):  
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, (party, party_lab) in enumerate(zip(parties, party_labels)):
        party_df = coefficients_df[(coefficients_df['model'] == model) 
                               & (coefficients_df['party'] == party)
                               & (coefficients_df['umich_var'] == umich_var)]
        if not party_df.empty:
            biden_1_beta_1 = party_df[party_df['biden'] == 1]['beta_1'].values[0]
            biden_0_beta_1 = party_df[party_df['biden'] == 0]['beta_1'].values[0]

            biden_1_beta_1_se = party_df[party_df['biden'] == 1]['beta_1_se'].values[0]
            biden_0_beta_1_se = party_df[party_df['biden'] == 0]['beta_1_se'].values[0]

            biden_1_beta_0 = party_df[party_df['biden'] == 1]['beta_0'].values[0]
            biden_0_beta_0 = party_df[party_df['biden'] == 0]['beta_0'].values[0]

            x_pos_biden_0_slope = i - 0.3
            x_pos_biden_1_slope = i - 0.1

            ax.bar(x_pos_biden_0_slope, biden_0_beta_1, 0.2, yerr=1.96 * biden_0_beta_1_se, 
                   color='r', alpha=0.6, label='Trump Presidency (slope)' if i == 0 else None)
            ax.bar(x_pos_biden_1_slope, biden_1_beta_1, 0.2, yerr=1.96 * biden_1_beta_1_se, 
                   color='b', alpha=0.6, label='Biden Presidency (slope)' if i == 0 else None)

            x_pos_biden_0_intercept = i + 0.1
            x_pos_biden_1_intercept = i + 0.3

            ax.bar(x_pos_biden_0_intercept, biden_0_beta_0, 0.2, yerr=1.96 * biden_0_beta_1_se, 
                   color='r', alpha=0.4, hatch='//', 
                   label='Trump Presidency (intercept)' if i == 0 else None)
            ax.bar(x_pos_biden_1_intercept, biden_1_beta_0, 0.2, yerr=1.96 * biden_1_beta_1_se, 
                   color='b', alpha=0.4, hatch='//', 
                   label='Biden Presidency (intercept)' if i == 0 else None)

    ax.set_xticks(np.arange(len(party_labels)))
    ax.set_xticklabels(party_labels)
    ax.set_ylabel('Point Estimate')
    ax.set_title(f'{umich_lab} Estimates by Partisan Affiliation for {model_lab}')
    ax.legend(loc='upper left')
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()

models=['monthly model', 'yoy model']
model_save_names=['monthly', 'yoy']
model_labs=['Monthly % Change Z-Score', 'YoY % Change Z-Score']
umich_vars = ['veh_r', 'hom_r', 'rinc_r', 'PSTK', 'newsrn_u_pri']
umich_labs=['Car-Buying Attitudes', 'Home-Buying Attitudes', 'Real Family Income Attitues',
            'Predicted Likelihood of Positive Stock Performance',
            'Reported Unfavorable News Coverage of Inflation']
umich_save_names=['car', 'hom', 'rinc', 'pstk', 'news_pri']
parties=['dem', 'rep', 'independent']
party_labels=['Democrats', 'Republicans', 'Independents']

for umich_var, umich_lab, umich_save_name in zip(umich_vars, umich_labs, umich_save_names):
    for model, model_lab, model_save_name in zip(models, model_labs, model_save_names):
        plot_regression_coefficients(coefficients_df, umich_var, umich_lab, model, model_lab, parties,
        party_labels, f'{macro_figures}/{model_save_name}_{umich_save_name}_party_comp.png')
# now let's run the same regressions but on an individual level
    # so now we're not recreating the '_r' variables but instead creating a series of dummies:
umich_filtered_df = pd.merge(umich_microdata_df, macro_var_df, on='date', how='left')

# now create our series of dummy variables:
for outcome, regressor in zip(['CAR', 'HOM', 'RINC'],
                              ['used car cpi', 'zillow index', 'real disposable income']):
    umich_filtered_df = umich_filtered_df[~umich_filtered_df[outcome].isin([8, 9])]
    for lab, num in zip(['good', 'poor'], [1,5]):
        umich_filtered_df[f'{outcome} {lab} dummy'] = (pd.to_numeric((umich_filtered_df[outcome] == num)
            .astype(int).fillna(0), errors='coerce'))

umich_filtered_df['news_72'] = ((umich_filtered_df['NEWS1'] == 72) 
                                 | (umich_filtered_df['NEWS2'] == 72).astype(int))

umich_vars = ['CAR good dummy', 'HOM good dummy', 'RINC good dummy', 'PSTK', 'news_72']
macro_vars = ['used car cpi', 'zillow index', 'real disposable income', 'sp500', 'headline cpi']

coefficients_micro_list = []
for outcome, regressor in zip(umich_vars, macro_vars):
    for party in ['dem', 'rep', 'independent']:
        for horizon, model_name in zip(['monthly', 'yoy'], ['monthly model', 'yoy model']):
            for biden_value, biden_descr in zip([1, 0], ['biden', 'pre-biden']):
                df_biden = umich_filtered_df[(umich_filtered_df[party] == 1) 
                                             & (umich_filtered_df['biden'] == biden_value)]
                run_and_store_regression(df_biden, outcome, 
                            f"{regressor} {horizon} pct. change zscore, {biden_descr}", party, 
                            biden_value, model_name, coefficients_micro_list)
coefficients_micro_df = pd.DataFrame(coefficients_micro_list, 
                               columns=[
                                   'model', 'umich_var', 'beta_0',
                                   'beta_1', 'party', 'beta_1_se', 'biden', 'sample size'
                                   ])
coefficients_micro_df.to_csv(f'{macro_figures}/coefficients_micro_table.csv', index=False)

models=['monthly model', 'yoy model']
model_save_names=['month', 'yoy']
model_labs=['Monthly % Change', 'YoY % Change']
umich_labs=['Positive Car-Buying Attitude', 'Positive Home-Buying Attitude',
            'Positive Real Family Income Attitude',
            'Predicted Likelihood of Positive Stock Performance',
            'Reported Unfavorable News Coverage of Inflation']
umich_save_names=['car', 'hom', 'rinc', 'pstk', 'news_pri']
parties=['dem', 'rep', 'independent']
party_labels=['Democrats', 'Republicans', 'Independents']

for umich_var, umich_lab, umich_save_name in zip(umich_vars, umich_labs, umich_save_names):
    for model, model_lab, model_save_name in zip(models, model_labs, model_save_names):
        plot_regression_coefficients(coefficients_micro_df, umich_var, umich_lab, model, model_lab, 
    parties, party_labels, f'{macro_figures}/{model_save_name}_{umich_save_name}_party_comp_micro.png')