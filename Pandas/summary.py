import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_datareader import data
from datetime import datetime
from dateutil import parser
from sqlalchemy import create_engine
import time

# ===============================
# 1. Data Structures
# ===============================
# Series examples
s = pd.Series([3, -5, 7, 4], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
pop_dict = {'California': 38332521, 'Texas': 26448193, 'New York': 19651127}
pop_series = pd.Series(pop_dict)

# DataFrame examples
df = pd.DataFrame({
    'Country': ['Belgium', 'India', 'Brazil'],
    'Capital': ['Brussels', 'New Delhi', 'Brasília'],
    'Population': [11190846, 1303171035, 207847528]
})
df_num = pd.DataFrame({"x": [0.123, 4.567, 8.901]})

movies = pd.DataFrame({
    "title": ["Avatar", "Avengers: Endgame", "Titanic", "Star Wars Ep. VII", "Avengers: Infinity War"],
    "release_year": [2009, 2019, 1997, 2015, 2018],
    "release_month": [12, 4, 11, 12, 4],
    "release_day": [18, 22, 1, 14, 23],
    "directors": ["James Cameron", "Anthony Russo, Joe Russo", "James Cameron", "J.J Abrams", "Anthony Russo, Joe Russo"],
    "box_office_busd": [2.922, 2.798, 2.202, 2.068, 2.048]
})

popcorn = pd.DataFrame({
    "brand": ["Orville", "PopIt"],
    "trial_1": [26, 14], "trial_2": [35, 34], "trial_3": [18, 21],
    "trial_4": [14, 37], "trial_5": [8, 29], "trial_6": [6, 23]
})

music = pd.DataFrame({
    "artist": ["Bad Bunny", "Drake"],
    "singles": [
        [{'title': 'Gato de Noche', 'tracks': [{'title': 'Gato de Noche', 'collaborator': 'Ñengo Flow'}]},
         {'title': 'La Jumpa', 'tracks': [{'title': 'La Jumpa', 'collaborator': 'Arcángel'}]}],
        [{'title': 'Scary Hours 2', 'tracks': [{'title': "What's Next"},
                                               {'title': 'Wants and Needs', 'collaborator': 'Lil Baby'},
                                               {'title': 'Lemon Pepper Freestyle', 'collaborator': 'Rick Ross'}]}]
    ]
})

people = pd.DataFrame({
    "sex": ["Female", "Male", "Female", "Male"],
    "hair_color": ["brown", "blonde", "black", "black"],
    "height_cm": [166, 184, 153, 192],
    "weight_kg": [72, None, 53, None]
})

# ===============================
# 2. Data Exploration & I/O
# ===============================
# DataFrame info
print(df.shape, df.index, df.columns)

# Save/load CSV and Excel
df.to_csv('myDataFrame.csv', index=False)
df_csv = pd.read_csv('myDataFrame.csv', nrows=5)
df.to_excel('myDataFrame.xlsx', sheet_name='Sheet1', index=False)
xls = pd.ExcelFile('myDataFrame.xlsx')
df_excel = pd.read_excel(xls, 'Sheet1')

# Save/load SQL
engine = create_engine('sqlite:///:memory:')
df.to_sql('myDf', engine, index=False)
df_sql = pd.read_sql_table('myDf', engine)

# ===============================
# 3. Selection & Indexing
# ===============================
# Element access
val_b = s2['b']
row_iloc = df.iloc[[0], [0]]
scalar_iat = df.iat[0, 0]
row_loc = df.loc[[0], ['Country']]
scalar_at = df.at[0, 'Country']
# Filter rows
filter_population = df[df['Population'] > 12000000]

# ===============================
# 4. Apply & Map Functions
# ===============================
f = lambda x: x * 2
df.apply(f)
df.applymap(f)

# ===============================
# 5. Arithmetic & Alignment
# ===============================
s3 = pd.Series([7, -2, 3], index=['a', 'c', 'd'])
add_series = s + s3
add_fill = s.add(s3, fill_value=0)
sub_fill = s.sub(s3, fill_value=2)
div_fill = s.div(s3, fill_value=4)
mul_fill = s.mul(s3, fill_value=3)

# ===============================
# 6. Drop & Null Handling
# ===============================
s_drop = s.drop(['a', 'c'])
df_drop = df.drop('Country', axis=1)
s_nan = pd.Series([1, np.nan, 3, None])
nulls = s_nan.isnull()
filled = s_nan.fillna(0)
ffill = s_nan.fillna(method='ffill')
bfill = s_nan.fillna(method='bfill')

# ===============================
# 7. Summary Statistics & Sorting
# ===============================
df_sum = df.sum()
df_cumsum = df.cumsum()
df_min = df.min()
df_max = df.max()
df_idxmin = df.idxmin()
df_idxmax = df.idxmax()
df_desc = df.describe()
df_mean = df.mean()
df_median = df.median()

# Sorting
df_sorted_idx = df.sort_index()
df_sorted_vals = df.sort_values(by='Sales')
df_sorted_vals = df.sort_values(by='Sales', ascending=False)
df_ranked = df.rank()

# ===============================
# 8. Boolean Indexing & Setting
# ===============================
s_mask_gt1 = s[s > 1]
s_mask_ltneg1_or_gt2 = s[(s < -1) | (s > 2)]
s['a'] = 6

# ===============================
# 9. String Operations
# ===============================
suits = pd.Series(["apple", "Banana", "cherry", "Date"])
contains_a = suits.str.contains("[ae]", case=False)
count_a = suits.str.count("[ae]")
find_e = suits.str.find("e")
find_all = suits.str.findall(".[ae]")
extract_groups = suits.str.extractall("([ae])(.)")
slice_2_5 = suits.str[2:5]
replace_a = suits.str.replace("a", "4", regex=True)
strip_s = suits.str.strip()
pad_s = suits.str.pad(8, fillchar='_')
lower = suits.str.lower()
upper = suits.str.upper()
title = suits.str.title()
capitalize = suits.str.capitalize()

# ===============================
# 10. Reshaping & Normalization
# ===============================
# Indexing
movies_idx = movies.set_index("title")
movies_reset = movies_idx.reset_index()

# Reindex
avengers = ["The Avengers", "Avengers: Age of Ultron", "Avengers: Infinity War", "Avengers: Endgame"]
movies_reindexed = movies_idx.reindex(avengers)

# Explode & Normalize
music_exploded = music.explode("singles")
music_norm = pd.json_normalize(music_exploded["singles"])

# Join & Split
movies['release_date'] = (movies['release_year'].astype(str) + "-" +
                          movies['release_month'].astype(str) + "-" +
                          movies['release_day'].astype(str))
movies[['director1', 'director2']] = movies['directors'].str.split(",", expand=True)

# Melt & Pivot
popcorn_melted = popcorn.melt(id_vars='brand', var_name='trial', value_name='n_unpopped')
popcorn_pivot = popcorn_melted.pivot(index='brand', columns='trial', values='n_unpopped')
popcorn_pivot_table = popcorn_melted.pivot_table(values='n_unpopped', index='brand', columns='trial')

# ===============================
# 11. Date & Time Operations
# ===============================
# Create datetime
dt = datetime(2015, 7, 4)
parsed = parser.parse("4th of July, 2015")
ts = pd.to_datetime("2015-07-04")
dates = pd.date_range('2015-07-03', '2015-07-10')
periods = pd.period_range('2015-07', periods=8, freq='M')
td_range = pd.timedelta_range(0, periods=10, freq='H')
np_date = np.array('2015-07-04', dtype='datetime64')
np_plus = np_date + np.arange(12)

# DataFrame with DateTimeIndex
dates_dt = pd.to_datetime([datetime(2015, 7, 3), '4th of July, 2015'])

# Resampling, shifting, rolling
goog = data.DataReader('GOOG', start='2004', end='2016', data_source='google')
close = goog['Close']
resampled_year_end = close.resample('BA').mean()
shifted = close.shift(900)
roi = 100 * (close.tshift(-365) / close - 1)
rolling_mean = close.rolling(365, center=True).mean()

# Plot stock
plt.figure(); close.plot(title='Google Close Price'); plt.show()
plt.figure(); roi.plot(title='ROI'); plt.show()

# Bicycle data mock
dates_bike = pd.date_range('2012-10-03', periods=100, freq='H')
bike_df = pd.DataFrame({
    'West': np.random.poisson(60, 100),
    'East': np.random.poisson(55, 100)
}, index=dates_bike)
bike_df['Total'] = bike_df['West'] + bike_df['East']
bike_df.resample('W').sum().plot(title='Weekly Bicycle Counts'); plt.show()
daily_bike = bike_df.resample('D').sum()
daily_bike.rolling(30, center=True).sum().plot(title='30-day Rolling Sum'); plt.show()
bike_df.groupby(bike_df.index.time).mean().plot(title='Avg Counts by Time'); plt.show()
bike_df.groupby(bike_df.index.dayofweek).mean().rename(index={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}).plot(title='Avg Counts by Weekday'); plt.show()

# Performance comparison
nrows, ncols = 100000, 100
df1 = pd.DataFrame(np.random.rand(nrows, ncols))
df2 = pd.DataFrame(np.random.rand(nrows, ncols))
start = time.time()
np.add(df1.values, df2.values)
print(f"Direct numpy sum: {time.time() - start:.2f}s")
start = time.time()
pd.eval('df1 + df2')
print(f"pd.eval sum: {time.time() - start:.2f}s")

# ===============================
# 12. Advanced String Operations
# ===============================
names = pd.Series(['peter', 'Paul', None, 'MARY', 'gUIDO'])
names_cap = names.str.capitalize()
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam', 'Eric Idle', 'Terry Jones', 'Michael Palin'])
lowercase = monte.str.lower()
lengths = monte.str.len()
starts_T = monte.str.startswith('T')
split_names = monte.str.split()
first_names = monte.str.extract('([A-Za-z]+)', expand=False)
last_names = monte.str.split().str.get(-1)
full_monte = pd.DataFrame({'name': monte, 'info': ['B|C|D', 'B|D', 'A|C', 'B|D', 'B|C', 'B|C|D']})
dummies = full_monte['info'].str.get_dummies('|')

# ===============================
# 13. JSON & Data Normalization
# ===============================
# Example JSON read omitted; dummy data:
recipes = pd.DataFrame({
    'name': ['Pancakes', 'Omelette'],
    'ingredients': ['flour|milk|egg', 'egg|milk|cheese'],
    'description': ['breakfast dish', 'quick meal']
})
recipes_exploded = recipes.explode('ingredients')
json_str = json.dumps(recipes['ingredients'].to_list())
recipes['ingredients2'] = json.loads(json_str)

# ===============================
# 14. Merge & Join
# ===============================
df_merge1 = pd.DataFrame({'employee': ['Bob', 'Jake'], 'group': ['A', 'B']})
df_merge2 = pd.DataFrame({'employee': ['Bob', 'Lisa'], 'hire_date': [2004, 2012]})
merged = pd.merge(df_merge1, df_merge2)
df_alt = pd.DataFrame({'name': ['Bob', 'Jake'], 'salary': [70000, 80000]})
merged_alt = pd.merge(df_merge1, df_alt, left_on='employee', right_on='name').drop('name', axis=1)
df_merge_idx1 = df_merge1.set_index('employee')
df_merge_idx2 = df_merge2.set_index('employee')
merged_idx = pd.merge(df_merge_idx1, df_merge_idx2, left_index=True, right_index=True)

# ===============================
# 15. Grouping & Aggregations
# ===============================
df_group = pd.DataFrame({'key': ['A','B','A','B'], 'value': [1,2,3,4]})
grouped = df_group.groupby('key').agg(['min', 'median', 'max'])
decade = pd.cut([2010, 2011, 2012, 2013], bins=range(2000, 2020, 10))

# ===============================
# 16. Plotting & Visualization
# ===============================
# Reusing Google stock data
goog['Close'].plot(title='Google Close Price'); plt.show()
# Bicycle counts
dates_bike = pd.date_range('2012-10-03', periods=100, freq='H')
bike_counts = pd.DataFrame({
    'West': np.random.poisson(60, 100),
    'East': np.random.poisson(55, 100)
}, index=dates_bike)
bike_counts['Total'] = bike_counts['West'] + bike_counts['East']
bike_counts.resample('W').sum().plot(title='Weekly Bicycle Counts'); plt.show()

# ===============================
# 17. Performance with pd.eval()
# ===============================
nrows, ncols = 1000, 100
df_big1 = pd.DataFrame(np.random.rand(nrows, ncols))
df_big2 = pd.DataFrame(np.random.rand(nrows, ncols))
start = time.time()
np.add(df_big1.values, df_big2.values)
print(f"Direct numpy sum: {time.time() - start:.2f}s")
start = time.time()
pd.eval('df_big1 + df_big2')
print(f"pd.eval sum: {time.time() - start:.2f}s")
