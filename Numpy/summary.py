# ============================================================
# NumPy Comprehensive Real-World Script (Optimized)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import special

sns.set()
np.random.seed(42)

# ============================================================
# 1. Array Making & Inspection
# ============================================================
arr1 = np.array([1,2,3,4,5])
arr2 = np.array([(1.5,2,3),(4,5,6)], dtype=float)
arr3 = np.array([[(1.5,2,3),(4,5,6)],[(3,2,1),(4,5,6)]], dtype=float)
zeros = np.zeros((3,4))
ones = np.ones((2,3,4), dtype=np.int16)
range_arr = np.arange(10,25,5)
linspace_arr = np.linspace(0,2,9)
full_arr = np.full((2,2),7)
eye_arr = np.eye(2)
rand_arr = np.random.random((2,2))
empty_arr = np.empty((3,2))

# ============================================================
# 2. Structured Arrays
# ============================================================
names = ['Alice','John','Cathy','Sandy']
ages = [25,45,37,19]
weights = [55.0,85.5,68.0,61.5]
structured = np.zeros(4, dtype={'names':('name','age','weight'),
                                'formats':('U10','i4','f8')})
structured['name'], structured['age'], structured['weight'] = names, ages, weights
structured_rec = structured.view(np.recarray)

# ============================================================
# 3. Indexing, Slicing, and Fancy Indexing
# ============================================================
arr = np.arange(10)
arr_slice = arr[2:7:2]
arr_rev = arr[::-1]
arr_fancy = arr[[1,3,5]]
mask = arr%2==0
arr[mask] = -1

# 2D example
X = np.arange(12).reshape((3,4))
row_idx, col_idx = [0,1,2], [2,1,3]
X_fancy = X[row_idx, col_idx]
X_broadcasted = X[row_idx[:,np.newaxis], col_idx]

# ============================================================
# 4. Arithmetic & Universal Functions
# ============================================================
x = np.arange(4)
y = x + 5
z = x**2
trig = np.sin(x)
log_val = np.log(x + 1)  # avoid log(0)
recip = 1.0 / (x + 1)
special_funcs = special.gamma([1,5,10])

# ============================================================
# 5. Aggregations & Statistics
# ============================================================
data = np.random.randn(1000)
aggregates = {
    'sum': np.sum(data),
    'mean': np.mean(data),
    'median': np.median(data),
    'std': np.std(data),
    'var': np.var(data),
    'min': np.min(data),
    'max': np.max(data),
    '90th_percentile': np.percentile(data,90)
}

# NaN-safe aggregation
x_nan = np.array([1,2,np.nan,4])
nan_agg = {
    'sum': np.nansum(x_nan),
    'mean': np.nanmean(x_nan)
}

# ============================================================
# 6. Broadcasting & Centering
# ============================================================
A = np.ones((3,3))
B = np.arange(3)
broadcast_sum = A + B
X_centered = data[:9].reshape((3,3)) - np.mean(data[:9].reshape((3,3)),0)

# ============================================================
# 7. Random Numbers & Sampling
# ============================================================
rand_uniform = np.random.rand(3,3)
rand_normal = np.random.randn(3,3)
rand_int = np.random.randint(0,10, size=(3,3))
sample_choice = np.random.choice([10,20,30,40], size=2, replace=False)
permutation = np.random.permutation(np.arange(10))

# ============================================================
# 8. Linear Algebra
# ============================================================
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
mat_mult = A @ B
det_A = np.linalg.det(A)
inv_A = np.linalg.inv(A)
eig_vals, eig_vecs = np.linalg.eig(A)

# ============================================================
# 9. Polynomials & FFT
# ============================================================
p = np.poly1d([1, -3, 2])
roots = p.r
val_at5 = p(5)

signal = np.random.rand(8)
fft_signal = np.fft.fft(signal)
ifft_signal = np.fft.ifft(fft_signal)

# ============================================================
# 10. Array Manipulation & Memory Efficiency
# ============================================================
arr_copy = arr.copy()
arr_view = arr.view()
arr_float = arr.astype(float)
arr *= 2  # in-place modification

# Reshaping, stacking, concatenation
reshaped = arr.reshape((2,5))
vstacked = np.vstack((arr, arr))
hstacked = np.hstack((arr[:5], arr[:5]))

# ============================================================
# 11. Sorting & Partial Sorting
# ============================================================
x_sort = np.array([7,2,3,1,6,5,4])
sorted_arr = np.sort(x_sort)
argsorted_arr = np.argsort(x_sort)
partitioned_arr = np.partition(x_sort, 3)

# ============================================================
# 12. Masking, Conditional Selection & np.where
# ============================================================
x = np.arange(10)
x[x%2==0] = -1
clipped = np.clip(x,0,5)
indices = np.where(x>2)
nonzero_vals = np.nonzero(x)

# ============================================================
# 13. File I/O
# ============================================================
np.save('my_array.npy', arr)
np.savez('array.npz', arr1=arr, arr2=arr_copy)
# np.load('my_array.npy')
# np.loadtxt("myfile.txt")
# np.genfromtxt("my_file.csv", delimiter=',')
# np.savetxt("myarray.txt", arr, delimiter=" ")

# ============================================================
# 14. String Operations
# ============================================================
names = np.array(['Alice','Bob','Cathy'])
name_lengths = np.char.str_len(names)
upper_names = np.char.upper(names)
