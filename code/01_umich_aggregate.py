import pandas as pd
from pathlib import Path
# import matplotlib.pyplot as plt
# let's create a set of locals referring to our directory and working directory 
home_dir = Path.home()
work_dir = (home_dir / 'consumer_sentiment')
data = (work_dir / 'data')
raw_data = (data / 'raw')
code = Path.cwd() 

umich_df = pd.read_csv(f'{raw_data}/umich_raw_all_tables.csv', header=1)

# let's do a mass renaming of variables according to the time series codebook (found in the 'resources' folder)
umich_df['day'] = 1
# Create a 'date' string column by concatenating 'year', 'month', and 'day'
umich_df['date_str'] = umich_df['yyyy'].astype(str) + '-' + umich_df['Month'].astype(str) + '-' + umich_df['day'].astype(str)
# Convert the 'date_str' column to a datetime coumn
umich_df['date'] = pd.to_datetime(umich_df['date_str'], format='%Y-%m-%d')

umich_df.to_csv(f'{data}/umich_aggregate_noparty.csv', index=False)