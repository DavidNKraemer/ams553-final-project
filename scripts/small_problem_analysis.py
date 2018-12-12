import pandas as pd

filename = '../data/small_problem_data_4_4_10.csv'

df = pd.read_csv(filename)

df['anneal_less_xentropy'] = df['anneal_cost'] - df['xentropy_cost']
print(df.groupby(['site_dist_id', 'drone_dist_id'])['anneal_less_xentropy'].mean())