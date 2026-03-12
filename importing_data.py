# ===============================
# Python Data Import Cheat Sheet
# ===============================

# Libraries
import numpy as np
import pandas as pd
import pickle
import os
import scipy.io
import h5py
from sas7bdat import SAS7BDAT
from sqlalchemy import create_engine


# -------------------------------
# 1. Importing Pickle Files
# -------------------------------
with open('pickled_fruit.pkl', 'rb') as file:
    pickled_data = pickle.load(file)

pickled_values = pickled_data.values()


# -------------------------------
# 2. Importing Plain Text Files
# -------------------------------
filename = "huck_finn.txt"

file = open(filename, mode='r')
text = file.read()
file.close()

with open(filename, 'r') as file:
    line1 = file.readline()
    line2 = file.readline()
    line3 = file.readline()


# -------------------------------
# 3. Importing Flat Files (NumPy)
# -------------------------------
data_loadtxt = np.loadtxt(
    'mnist.txt',
    delimiter=',',
    skiprows=2,
    usecols=[0, 2],
    dtype=str
)

data_genfromtxt = np.genfromtxt(
    'titanic.csv',
    delimiter=',',
    names=True,
    dtype=None
)

data_recfromcsv = np.recfromcsv('titanic.csv')


# -------------------------------
# 4. Importing Flat Files (Pandas)
# -------------------------------
df_csv = pd.read_csv(
    'winequality-red.csv',
    nrows=5,
    header=None,
    sep='\t',
    comment='#',
    na_values=[""]
)

data_array = df_csv.values


# -------------------------------
# 5. Exploring Pandas DataFrame
# -------------------------------
head_rows = df_csv.head()
tail_rows = df_csv.tail()
df_index = df_csv.index
df_columns = df_csv.columns
df_info = df_csv.info()


# -------------------------------
# 6. Importing Excel Files
# -------------------------------
excel_file = 'urbanpop.xlsx'
data = pd.ExcelFile(excel_file)

df_sheet1 = data.parse(
    0,
    parse_cols=[0],
    skiprows=[0],
    names=['Country']
)

df_sheet2 = data.parse(
    '1960-1966',
    skiprows=[0],
    names=['Country', 'AAM: War(2002)']
)

sheet_names = data.sheet_names


# -------------------------------
# 7. Importing SAS Files
# -------------------------------
with SAS7BDAT('urbanpop.sas7bdat') as file:
    df_sas = file.to_data_frame()


# -------------------------------
# 8. Importing Stata Files
# -------------------------------
df_stata = pd.read_stata('urbanpop.dta')


# -------------------------------
# 9. Importing MATLAB Files
# -------------------------------
mat = scipy.io.loadmat('workspace.mat')

mat_keys = mat.keys()
mat_items = mat.items()


# -------------------------------
# 10. Importing HDF5 Files
# -------------------------------
filename = 'H-H1_LOSC_4_v1-815411200-4096.hdf5'
data_hdf5 = h5py.File(filename, 'r')

keys = data_hdf5.keys()

meta_keys = data_hdf5['meta'].keys()

description = data_hdf5['meta']['Description'][()]


# -------------------------------
# 11. Relational Databases (SQL)
# -------------------------------
engine = create_engine('sqlite:///Northwind.sqlite')

table_names = engine.table_names()

con = engine.connect()
rs = con.execute("SELECT * FROM Orders")

df_sql = pd.DataFrame(rs.fetchall())
df_sql.columns = rs.keys()

con.close()

df_sql_query = pd.read_sql_query("SELECT * FROM Orders", engine)

with engine.connect() as con:
    rs = con.execute("SELECT OrderID FROM Orders")
    df_limited = pd.DataFrame(rs.fetchmany(size=5))
    df_limited.columns = rs.keys()


# -------------------------------
# 12. File System Navigation
# -------------------------------
wd = os.getcwd()
files = os.listdir(wd)

os.chdir("/usr/tmp")

os.rename("test1.txt", "test2.txt")
os.remove("test1.txt")

os.mkdir("newdir")
