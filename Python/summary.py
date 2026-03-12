# Ultimate Python & Data Science Cheat Sheet
import pandas as pd
import numpy as np
import os
from math import pi

# -------------------------
# 1. Variables & Calculations
# -------------------------
x = 5
y = 2
sum_xy = x + y
diff_xy = x - y
prod_xy = x * y
exp_x = x ** 2
mod_x = x % 2
div_x = x / float(2)

# Arithmetic operators
sum_val = 102 + 37
diff_val = 102 - 37
prod_val = 4 * 6
div_val = 22 / 7
int_div = 22 // 7
mod_val = 22 % 7
pow_val = 3 ** 4

# Comparisons
eq = 3 == 3
neq = 3 != 3
gt = 3 > 1
gte = 3 >= 3
lt = 3 < 4
lte = 3 <= 4

# Logical
logical_not = not(2 == 2)
logical_and = (1 != 1) and (1 < 1)
logical_or = (1 >= 1) or (1 < 1)

# -------------------------
# 2. Strings
# -------------------------
my_string = "thisStringIsAwesomeInnit"
char_index3 = my_string[3]
slice_4_9 = my_string[4:9]
uppercase_str = my_string.upper()
lowercase_str = my_string.lower()
count_w = my_string.count('w')
replace_e_i = my_string.replace('e', 'i')
strip_str = my_string.strip()
contains_m = 'm' in my_string

# String operations
a_str = 'is'
b_str = 'nice'
my_list_str = ['my', 'list', a_str, b_str]
s = "Jack and Jill"
s_upper = s.upper()
s_lower = s.lower()
s_title = s.title()
s_replace = s.replace("J", "P")
s_split_e = "beekeepers".split("e")
s_first_char = s[0]
s_substring = s[0:2]
s_concat = "Data" + "Framed"
s_repeat = 3 * "data "
s_quote = "He said, \"DataCamp\""

# -------------------------
# 3. Lists
# -------------------------
x_list = [1, 3, 6]
y_list = [10, 15, 21]

# List operations
x_sorted = sorted(x_list)
x_list.sort()
x_reversed = list(reversed(x_list))
x_count_2 = x_list.count(2)
x_concat_y = x_list + y_list
x_repeat = 3 * x_list
x_slice1 = x_list[1:3]
x_slice2 = x_list[2:]
x_slice3 = x_list[:3]

# Nested lists
my_list2 = [[4,5,6,7], [3,4,5,6]]
nested_elem = my_list2[1][0]
nested_slice = my_list2[1][:2]

# List methods
index_a = my_list_str.index(a_str)
count_a = my_list_str.count(a_str)
my_list_str.append('!')
my_list_str.remove('!')
del(my_list_str[0:1])
my_list_str.reverse()
my_list_str.extend('!')
popped = my_list_str.pop(-1)
my_list_str.insert(0,'!')
my_list_str.sort()

# List operations
concat_list = my_list_str + my_list_str
repeat_list = my_list_str * 2

# -------------------------
# 4. Dictionaries
# -------------------------
x_dict = {'a': 1, 'b': 2, 'c': 3}
x_dict_values = list(x_dict.values())
x_dict_keys = list(x_dict.keys())
a_value = x_dict['a']

# -------------------------
# 5. NumPy Arrays
# -------------------------
arr = np.array([1, 2, 3])
arr_range = np.arange(1, 5)
arr_range_step = np.arange(1, 5, 2)
arr_repeat = np.repeat([1, 3, 6], 3)
arr_tile = np.tile([1, 3, 6], 3)

# Array operations
my_array = np.array([1, 2, 3, 4])
my_2darray = np.array([[1,2,3],[4,5,6]])
elem_1 = my_array[1]
slice_0_2 = my_array[0:2]
col_0 = my_2darray[:,0]
greater_3 = my_array > 3
double_array = my_array * 2
sum_array = my_array + np.array([5,6,7,8])

# Array functions
shape = my_array.shape
append_array = np.append(my_array, [5,6])
insert_array = np.insert(my_array, 1, 5)
delete_array = np.delete(my_array, [1])
mean_array = np.mean(my_array)
median_array = np.median(my_array)
std_array = np.std(my_array)
# corr_coef = np.corrcoef(my_array, other_array)  # Example correlation

# -------------------------
# 6. Pandas DataFrames
# -------------------------
df = pd.DataFrame([
    {'a': 1, 'b': 4, 'c': 'x'},
    {'a': 1, 'b': 4, 'c': 'x'},
    {'a': 3, 'b': 6, 'c': 'y'}
])

# Selection
row3 = df.iloc[2]
col_c = df['c']
cols_a_b = df[['a', 'b']]
col_pos2 = df.iloc[:, 2]
element_row3_col2 = df.iloc[2, 1]

# Manipulation
df_concat_vert = pd.concat([df, df])
df_concat_horiz = pd.concat([df, df], axis="columns")
df_query = df.query('b > 4')
df_drop_col = df.drop(columns=['c'])
df_rename = df.rename(columns={"a": "alpha"})
df_new_col = df.assign(temp_f=9 / 5 * df['b'] + 32)
df_mean = df.mean(numeric_only=True)
df_agg = df.agg(['sum', 'mean'])
df_unique = df.drop_duplicates()
df_sort = df.sort_values(by='b')
df_nlargest = df.nlargest(1, 'b')

# -------------------------
# 7. Packages & Working Directory
# -------------------------
cwd = os.getcwd()
# os.chdir("new/working/directory")  # Set working directory

# -------------------------
# 8. Accessing Help & Object Types
# -------------------------
max_help = help(max)
type_str = type('a')
