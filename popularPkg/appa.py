
# coding: utf-8

# # Advanced NumPy

# In[ ]:


import numpy as np
import pandas as pd
np.random.seed(12345)
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(10, 6))
PREVIOUS_MAX_ROWS = pd.options.display.max_rows
pd.options.display.max_rows = 20
np.set_printoptions(precision=4, suppress=True)


# ## ndarray Object Internals

# In[ ]:


np.ones((10, 5)).shape


# In[ ]:


np.ones((3, 4, 5), dtype=np.float64).strides


# ### NumPy dtype Hierarchy

# In[ ]:


ints = np.ones(10, dtype=np.uint16)
floats = np.ones(10, dtype=np.float32)
np.issubdtype(ints.dtype, np.integer)
np.issubdtype(floats.dtype, np.floating)


# In[ ]:


np.float64.mro()


# In[ ]:


np.issubdtype(ints.dtype, np.number)


# ## Advanced Array Manipulation

# ### Reshaping Arrays

# In[ ]:


arr = np.arange(8)
arr
arr.reshape((4, 2))


# In[ ]:


arr.reshape((4, 2)).reshape((2, 4))


# In[ ]:


arr = np.arange(15)
arr.reshape((5, -1))


# In[ ]:


other_arr = np.ones((3, 5))
other_arr.shape
arr.reshape(other_arr.shape)


# In[ ]:


arr = np.arange(15).reshape((5, 3))
arr
arr.ravel()


# In[ ]:


arr.flatten()


# ### C Versus Fortran Order

# In[ ]:


arr = np.arange(12).reshape((3, 4))
arr
arr.ravel()
arr.ravel('F')


# ### Concatenating and Splitting Arrays

# In[ ]:


arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
np.concatenate([arr1, arr2], axis=0)
np.concatenate([arr1, arr2], axis=1)


# In[ ]:


np.vstack((arr1, arr2))
np.hstack((arr1, arr2))


# In[ ]:


arr = np.random.randn(5, 2)
arr
first, second, third = np.split(arr, [1, 3])
first
second
third


# #### Stacking helpers: r_ and c_

# In[ ]:


arr = np.arange(6)
arr1 = arr.reshape((3, 2))
arr2 = np.random.randn(3, 2)
np.r_[arr1, arr2]
np.c_[np.r_[arr1, arr2], arr]


# In[ ]:


np.c_[1:6, -10:-5]


# ### Repeating Elements: tile and repeat

# In[ ]:


arr = np.arange(3)
arr
arr.repeat(3)


# In[ ]:


arr.repeat([2, 3, 4])


# In[ ]:


arr = np.random.randn(2, 2)
arr
arr.repeat(2, axis=0)


# In[ ]:


arr.repeat([2, 3], axis=0)
arr.repeat([2, 3], axis=1)


# In[ ]:


arr
np.tile(arr, 2)


# In[ ]:


arr
np.tile(arr, (2, 1))
np.tile(arr, (3, 2))


# ### Fancy Indexing Equivalents: take and put

# In[ ]:


arr = np.arange(10) * 100
inds = [7, 1, 2, 6]
arr[inds]


# In[ ]:


arr.take(inds)
arr.put(inds, 42)
arr
arr.put(inds, [40, 41, 42, 43])
arr


# In[ ]:


inds = [2, 0, 2, 1]
arr = np.random.randn(2, 4)
arr
arr.take(inds, axis=1)


# ## Broadcasting

# In[ ]:


arr = np.arange(5)
arr
arr * 4


# In[ ]:


arr = np.random.randn(4, 3)
arr.mean(0)
demeaned = arr - arr.mean(0)
demeaned
demeaned.mean(0)


# In[ ]:


arr
row_means = arr.mean(1)
row_means.shape
row_means.reshape((4, 1))
demeaned = arr - row_means.reshape((4, 1))
demeaned.mean(1)


# ### Broadcasting Over Other Axes

# In[ ]:


arr - arr.mean(1)


# In[ ]:


arr - arr.mean(1).reshape((4, 1))


# In[ ]:


arr = np.zeros((4, 4))
arr_3d = arr[:, np.newaxis, :]
arr_3d.shape
arr_1d = np.random.normal(size=3)
arr_1d[:, np.newaxis]
arr_1d[np.newaxis, :]


# In[ ]:


arr = np.random.randn(3, 4, 5)
depth_means = arr.mean(2)
depth_means
depth_means.shape
demeaned = arr - depth_means[:, :, np.newaxis]
demeaned.mean(2)


# ```python
# def demean_axis(arr, axis=0):
#     means = arr.mean(axis)
# 
#     # This generalizes things like [:, :, np.newaxis] to N dimensions
#     indexer = [slice(None)] * arr.ndim
#     indexer[axis] = np.newaxis
#     return arr - means[indexer]
# ```

# ### Setting Array Values by Broadcasting

# In[ ]:


arr = np.zeros((4, 3))
arr[:] = 5
arr


# In[ ]:


col = np.array([1.28, -0.42, 0.44, 1.6])
arr[:] = col[:, np.newaxis]
arr
arr[:2] = [[-1.37], [0.509]]
arr


# ## Advanced ufunc Usage

# ### ufunc Instance Methods

# In[ ]:


arr = np.arange(10)
np.add.reduce(arr)
arr.sum()


# In[ ]:


np.random.seed(12346)  # for reproducibility
arr = np.random.randn(5, 5)
arr[::2].sort(1) # sort a few rows
arr[:, :-1] < arr[:, 1:]
np.logical_and.reduce(arr[:, :-1] < arr[:, 1:], axis=1)


# In[ ]:


arr = np.arange(15).reshape((3, 5))
np.add.accumulate(arr, axis=1)


# In[ ]:


arr = np.arange(3).repeat([1, 2, 2])
arr
np.multiply.outer(arr, np.arange(5))


# In[ ]:


x, y = np.random.randn(3, 4), np.random.randn(5)
result = np.subtract.outer(x, y)
result.shape


# In[ ]:


arr = np.arange(10)
np.add.reduceat(arr, [0, 5, 8])


# In[ ]:


arr = np.multiply.outer(np.arange(4), np.arange(5))
arr
np.add.reduceat(arr, [0, 2, 4], axis=1)


# ### Writing New ufuncs in Python

# In[ ]:


def add_elements(x, y):
    return x + y
add_them = np.frompyfunc(add_elements, 2, 1)
add_them(np.arange(8), np.arange(8))


# In[ ]:


add_them = np.vectorize(add_elements, otypes=[np.float64])
add_them(np.arange(8), np.arange(8))


# In[ ]:


arr = np.random.randn(10000)
get_ipython().magic('timeit add_them(arr, arr)')
get_ipython().magic('timeit np.add(arr, arr)')


# ## Structured and Record Arrays

# In[ ]:


dtype = [('x', np.float64), ('y', np.int32)]
sarr = np.array([(1.5, 6), (np.pi, -2)], dtype=dtype)
sarr


# In[ ]:


sarr[0]
sarr[0]['y']


# In[ ]:


sarr['x']


# ### Nested dtypes and Multidimensional Fields

# In[ ]:


dtype = [('x', np.int64, 3), ('y', np.int32)]
arr = np.zeros(4, dtype=dtype)
arr


# In[ ]:


arr[0]['x']


# In[ ]:


arr['x']


# In[ ]:


dtype = [('x', [('a', 'f8'), ('b', 'f4')]), ('y', np.int32)]
data = np.array([((1, 2), 5), ((3, 4), 6)], dtype=dtype)
data['x']
data['y']
data['x']['a']


# ### Why Use Structured Arrays?

# ## More About Sorting

# In[ ]:


arr = np.random.randn(6)
arr.sort()
arr


# In[ ]:


arr = np.random.randn(3, 5)
arr
arr[:, 0].sort()  # Sort first column values in-place
arr


# In[ ]:


arr = np.random.randn(5)
arr
np.sort(arr)
arr


# In[ ]:


arr = np.random.randn(3, 5)
arr
arr.sort(axis=1)
arr


# In[ ]:


arr[:, ::-1]


# ### Indirect Sorts: argsort and lexsort

# In[ ]:


values = np.array([5, 0, 1, 3, 2])
indexer = values.argsort()
indexer
values[indexer]


# In[ ]:


arr = np.random.randn(3, 5)
arr[0] = values
arr
arr[:, arr[0].argsort()]


# In[ ]:


first_name = np.array(['Bob', 'Jane', 'Steve', 'Bill', 'Barbara'])
last_name = np.array(['Jones', 'Arnold', 'Arnold', 'Jones', 'Walters'])
sorter = np.lexsort((first_name, last_name))
sorter
zip(last_name[sorter], first_name[sorter])


# ### Alternative Sort Algorithms

# In[ ]:


values = np.array(['2:first', '2:second', '1:first', '1:second',
                   '1:third'])
key = np.array([2, 2, 1, 1, 1])
indexer = key.argsort(kind='mergesort')
indexer
values.take(indexer)


# ### Partially Sorting Arrays

# In[ ]:


np.random.seed(12345)
arr = np.random.randn(20)
arr
np.partition(arr, 3)


# In[ ]:


indices = np.argpartition(arr, 3)
indices
arr.take(indices)


# ### numpy.searchsorted: Finding Elements in a Sorted Array

# In[ ]:


arr = np.array([0, 1, 7, 12, 15])
arr.searchsorted(9)


# In[ ]:


arr.searchsorted([0, 8, 11, 16])


# In[ ]:


arr = np.array([0, 0, 0, 1, 1, 1, 1])
arr.searchsorted([0, 1])
arr.searchsorted([0, 1], side='right')


# In[ ]:


data = np.floor(np.random.uniform(0, 10000, size=50))
bins = np.array([0, 100, 1000, 5000, 10000])
data


# In[ ]:


labels = bins.searchsorted(data)
labels


# In[ ]:


pd.Series(data).groupby(labels).mean()


# ## Writing Fast NumPy Functions with Numba

# In[ ]:


import numpy as np

def mean_distance(x, y):
    nx = len(x)
    result = 0.0
    count = 0
    for i in range(nx):
        result += x[i] - y[i]
        count += 1
    return result / count


# ```python
# In [209]: x = np.random.randn(10000000)
# 
# In [210]: y = np.random.randn(10000000)
# 
# In [211]: %timeit mean_distance(x, y)
# 1 loop, best of 3: 2 s per loop
# 
# In [212]: %timeit (x - y).mean()
# 100 loops, best of 3: 14.7 ms per loop
# ```

# ```python
# In [213]: import numba as nb
# 
# In [214]: numba_mean_distance = nb.jit(mean_distance)
# ```

# ```python
# @nb.jit
# def mean_distance(x, y):
#     nx = len(x)
#     result = 0.0
#     count = 0
#     for i in range(nx):
#         result += x[i] - y[i]
#         count += 1
#     return result / count
# ```

# ```python
# In [215]: %timeit numba_mean_distance(x, y)
# 100 loops, best of 3: 10.3 ms per loop
# ```

# ```python
# from numba import float64, njit
# 
# @njit(float64(float64[:], float64[:]))
# def mean_distance(x, y):
#     return (x - y).mean()
# ```

# ### Creating Custom numpy.ufunc Objects with Numba

# ```python
# from numba import vectorize
# 
# @vectorize
# def nb_add(x, y):
#     return x + y
# ```

# ```python
# In [13]: x = np.arange(10)
# 
# In [14]: nb_add(x, x)
# Out[14]: array([  0.,   2.,   4.,   6.,   8.,  10.,  12.,  14.,  16.,  18.])
# 
# In [15]: nb_add.accumulate(x, 0)
# Out[15]: array([  0.,   1.,   3.,   6.,  10.,  15.,  21.,  28.,  36.,  45.])
# ```

# ## Advanced Array Input and Output

# ### Memory-Mapped Files

# In[ ]:


mmap = np.memmap('mymmap', dtype='float64', mode='w+',
                 shape=(10000, 10000))
mmap


# In[ ]:


section = mmap[:5]


# In[ ]:


section[:] = np.random.randn(5, 10000)
mmap.flush()
mmap
del mmap


# In[ ]:


mmap = np.memmap('mymmap', dtype='float64', shape=(10000, 10000))
mmap


# In[ ]:


get_ipython().magic('xdel mmap')
get_ipython().system('rm mymmap')


# ### HDF5 and Other Array Storage Options

# ## Performance Tips

# ### The Importance of Contiguous Memory

# In[ ]:


arr_c = np.ones((1000, 1000), order='C')
arr_f = np.ones((1000, 1000), order='F')
arr_c.flags
arr_f.flags
arr_f.flags.f_contiguous


# In[ ]:


get_ipython().magic('timeit arr_c.sum(1)')
get_ipython().magic('timeit arr_f.sum(1)')


# In[ ]:


arr_f.copy('C').flags


# In[ ]:


arr_c[:50].flags.contiguous
arr_c[:, :50].flags


# In[ ]:


get_ipython().magic('xdel arr_c')
get_ipython().magic('xdel arr_f')


# In[ ]:


pd.options.display.max_rows = PREVIOUS_MAX_ROWS

