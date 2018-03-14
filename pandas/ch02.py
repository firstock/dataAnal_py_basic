
# coding: utf-8

# # Python Language Basics, IPython, and Jupyter Notebooks

# In[ ]:


import numpy as np
np.random.seed(12345)
np.set_printoptions(precision=4, suppress=True)


# ## The Python Interpreter

# ```python
# $ python
# Python 3.6.0 | packaged by conda-forge | (default, Jan 13 2017, 23:17:12)
# [GCC 4.8.2 20140120 (Red Hat 4.8.2-15)] on linux
# Type "help", "copyright", "credits" or "license" for more information.
# >>> a = 5
# >>> print(a)
# 5
# ```

# ```python
# print('Hello world')
# ```

# ```python
# $ python hello_world.py
# Hello world
# ```

# ```shell
# $ ipython
# Python 3.6.0 | packaged by conda-forge | (default, Jan 13 2017, 23:17:12)
# Type "copyright", "credits" or "license" for more information.
# 
# IPython 5.1.0 -- An enhanced Interactive Python.
# ?         -> Introduction and overview of IPython's features.
# %quickref -> Quick reference.
# help      -> Python's own help system.
# object?   -> Details about 'object', use 'object??' for extra details.
# 
# In [1]: %run hello_world.py
# Hello world
# 
# In [2]:
# ```

# ## IPython Basics

# ### Running the IPython Shell

# $ 

# In[1]:


import numpy as np
data = {i : np.random.randn() for i in range(7)}
data


# >>> from numpy.random import randn
# >>> data = {i : randn() for i in range(7)}
# >>> print(data)
# {0: -1.5948255432744511, 1: 0.10569006472787983, 2: 1.972367135977295,
# 3: 0.15455217573074576, 4: -0.24058577449429575, 5: -1.2904897053651216,
# 6: 0.3308507317325902}

# ### Running the Jupyter Notebook

# ```shell
# $ jupyter notebook
# [I 15:20:52.739 NotebookApp] Serving notebooks from local directory:
# /home/wesm/code/pydata-book
# [I 15:20:52.739 NotebookApp] 0 active kernels
# [I 15:20:52.739 NotebookApp] The Jupyter Notebook is running at:
# http://localhost:8888/
# [I 15:20:52.740 NotebookApp] Use Control-C to stop this server and shut down
# all kernels (twice to skip confirmation).
# Created new window in existing browser session.
# ```

# ### Tab Completion

# ```
# In [1]: an_apple = 27
# 
# In [2]: an_example = 42
# 
# In [3]: an
# ```

# ```
# In [3]: b = [1, 2, 3]
# 
# In [4]: b.
# ```

# ```
# In [1]: import datetime
# 
# In [2]: datetime.
# ```

# ```
# In [7]: datasets/movielens/
# ```

# ### Introspection

# ```
# In [8]: b = [1, 2, 3]
# 
# In [9]: b?
# Type:       list
# String Form:[1, 2, 3]
# Length:     3
# Docstring:
# list() -> new empty list
# list(iterable) -> new list initialized from iterable's items
# 
# In [10]: print?
# Docstring:
# print(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False)
# 
# Prints the values to a stream, or to sys.stdout by default.
# Optional keyword arguments:
# file:  a file-like object (stream); defaults to the current sys.stdout.
# sep:   string inserted between values, default a space.
# end:   string appended after the last value, default a newline.
# flush: whether to forcibly flush the stream.
# Type:      builtin_function_or_method
# ```

# ```python
# def add_numbers(a, b):
#     """
#     Add two numbers together
# 
#     Returns
#     -------
#     the_sum : type of arguments
#     """
#     return a + b
# ```

# ```python
# In [11]: add_numbers?
# Signature: add_numbers(a, b)
# Docstring:
# Add two numbers together
# 
# Returns
# -------
# the_sum : type of arguments
# File:      <ipython-input-9-6a548a216e27>
# Type:      function
# ```

# ```python
# In [12]: add_numbers??
# Signature: add_numbers(a, b)
# Source:
# def add_numbers(a, b):
#     """
#     Add two numbers together
# 
#     Returns
#     -------
#     the_sum : type of arguments
#     """
#     return a + b
# File:      <ipython-input-9-6a548a216e27>
# Type:      function
# ```

# ```python
# In [13]: np.*load*?
# np.__loader__
# np.load
# np.loads
# np.loadtxt
# np.pkgload
# ```

# ### The %run Command

# ```python
# def f(x, y, z):
#     return (x + y) / z
# 
# a = 5
# b = 6
# c = 7.5
# 
# result = f(a, b, c)
# ```

# ```python
# In [14]: %run ipython_script_test.py
# ```

# ```python
# In [15]: c
# Out [15]: 7.5
# 
# In [16]: result
# Out[16]: 1.4666666666666666
# ```

# ```python
# >>> %load ipython_script_test.py
# 
#     def f(x, y, z):
#         return (x + y) / z
# 
#     a = 5
#     b = 6
#     c = 7.5
# 
#     result = f(a, b, c)
# ```

# #### Interrupting running code

# ### Executing Code from the Clipboard

# ```python
# x = 5
# y = 7
# if x > 5:
#     x += 1
# 
#     y = 8
# ```

# ```python
# In [17]: %paste
# x = 5
# y = 7
# if x > 5:
#     x += 1
# 
#     y = 8
# ## -- End pasted text --
# ```

# ```python
# In [18]: %cpaste
# Pasting code; enter '--' alone on the line to stop or use Ctrl-D.
# :x = 5
# :y = 7
# :if x > 5:
# :    x += 1
# :
# :    y = 8
# :--
# ```

# ### Terminal Keyboard Shortcuts

# ### About Magic Commands

# ```python
# In [20]: a = np.random.randn(100, 100)
# 
# In [20]: %timeit np.dot(a, a)
# 10000 loops, best of 3: 20.9 µs per loop
# ```

# ```python
# In [21]: %debug?
# Docstring:
# ::
# 
#   %debug [--breakpoint FILE:LINE] [statement [statement ...]]
# 
# Activate the interactive debugger.
# 
# This magic command support two ways of activating debugger.
# One is to activate debugger before executing code.  This way, you
# can set a break point, to step through the code from the point.
# You can use this mode by giving statements to execute and optionally
# a breakpoint.
# 
# The other one is to activate debugger in post-mortem mode.  You can
# activate this mode simply running %debug without any argument.
# If an exception has just occurred, this lets you inspect its stack
# frames interactively.  Note that this will always work only on the last
# traceback that occurred, so you must call this quickly after an
# exception that you wish to inspect has fired, because if another one
# occurs, it clobbers the previous one.
# 
# If you want IPython to automatically do this on every exception, see
# the %pdb magic for more details.
# 
# positional arguments:
#   statement             Code to run in debugger. You can omit this in cell
#                         magic mode.
# 
# optional arguments:
#   --breakpoint <FILE:LINE>, -b <FILE:LINE>
#                         Set break point at LINE in FILE.
# 
# ```                        

# ```python
# In [22]: %pwd
# Out[22]: '/home/wesm/code/pydata-book
# 
# In [23]: foo = %pwd
# 
# In [24]: foo
# Out[24]: '/home/wesm/code/pydata-book'
# ```

# ### Matplotlib Integration

# ```python
# In [26]: %matplotlib
# Using matplotlib backend: Qt4Agg
# ```

# ```python
# In [26]: %matplotlib inline
# ```

# ## Python Language Basics

# ### Language Semantics

# #### Indentation, not braces

# ```python
# for x in array:
#     if x < pivot:
#         less.append(x)
#     else:
#         greater.append(x)
# ```

# ```python
# a = 5; b = 6; c = 7
# ```

# #### Everything is an object

# #### Comments

# ```python
# results = []
# for line in file_handle:
#     # keep the empty lines for now
#     # if len(line) == 0:
#     #   continue
#     results.append(line.replace('foo', 'bar'))
# ```

# ```python
# print("Reached this line")  # Simple status report
# ```

# #### Function and object method calls

# ```
# result = f(x, y, z)
# g()
# ```

# ```
# obj.some_method(x, y, z)
# ```

# ```python
# result = f(a, b, c, d=5, e='foo')
# ```

# #### Variables and argument passing

# In[ ]:


a = [1, 2, 3]


# In[ ]:


b = a


# In[ ]:


a.append(4)
b


# ```python
# def append_element(some_list, element):
#     some_list.append(element)
# ```

# ```python
# In [27]: data = [1, 2, 3]
# 
# In [28]: append_element(data, 4)
# 
# In [29]: data
# Out[29]: [1, 2, 3, 4]
# ```

# #### Dynamic references, strong types

# In[ ]:


a = 5
type(a)
a = 'foo'
type(a)


# In[ ]:


'5' + 5


# In[ ]:


a = 4.5
b = 2
# String formatting, to be visited later
print('a is {0}, b is {1}'.format(type(a), type(b)))
a / b


# In[ ]:


a = 5
isinstance(a, int)


# In[ ]:


a = 5; b = 4.5
isinstance(a, (int, float))
isinstance(b, (int, float))


# #### Attributes and methods

# ```python
# In [1]: a = 'foo'
# 
# In [2]: a.<Press Tab>
# a.capitalize  a.format      a.isupper     a.rindex      a.strip
# a.center      a.index       a.join        a.rjust       a.swapcase
# a.count       a.isalnum     a.ljust       a.rpartition  a.title
# a.decode      a.isalpha     a.lower       a.rsplit      a.translate
# a.encode      a.isdigit     a.lstrip      a.rstrip      a.upper
# a.endswith    a.islower     a.partition   a.split       a.zfill
# a.expandtabs  a.isspace     a.replace     a.splitlines
# a.find        a.istitle     a.rfind       a.startswith
# ```

# In[ ]:


a = 'foo'


# In[ ]:


getattr(a, 'split')


# #### Duck typing

# In[ ]:


def isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError: # not iterable
        return False


# In[ ]:


isiterable('a string')
isiterable([1, 2, 3])
isiterable(5)


# if not isinstance(x, list) and isiterable(x):
#     x = list(x)

# #### Imports

# ```python
# # some_module.py
# PI = 3.14159
# 
# def f(x):
#     return x + 2
# 
# def g(a, b):
#     return a + b
# ```

# import some_module
# result = some_module.f(5)
# pi = some_module.PI

# from some_module import f, g, PI
# result = g(5, PI)

# import some_module as sm
# from some_module import PI as pi, g as gf
# 
# r1 = sm.f(pi)
# r2 = gf(6, pi)

# #### Binary operators and comparisons

# In[ ]:


5 - 7
12 + 21.5
5 <= 2


# In[ ]:


a = [1, 2, 3]
b = a
c = list(a)
a is b
a is not c


# In[ ]:


a == c


# In[ ]:


a = None
a is None


# #### Mutable and immutable objects

# In[ ]:


a_list = ['foo', 2, [4, 5]]
a_list[2] = (3, 4)
a_list


# In[ ]:


a_tuple = (3, 5, (4, 5))
a_tuple[1] = 'four'


# ### Scalar Types

# #### Numeric types

# In[ ]:


ival = 17239871
ival ** 6


# In[ ]:


fval = 7.243
fval2 = 6.78e-5


# In[ ]:


3 / 2


# In[ ]:


3 // 2


# #### Strings

# a = 'one way of writing a string'
# b = "another way"

# In[ ]:


c = """
This is a longer string that
spans multiple lines
"""


# In[ ]:


c.count('\n')


# In[ ]:


a = 'this is a string'
a[10] = 'f'
b = a.replace('string', 'longer string')
b


# In[ ]:


a


# In[ ]:


a = 5.6
s = str(a)
print(s)


# In[ ]:


s = 'python'
list(s)
s[:3]


# In[ ]:


s = '12\\34'
print(s)


# In[ ]:


s = r'this\has\no\special\characters'
s


# In[ ]:


a = 'this is the first half '
b = 'and this is the second half'
a + b


# In[ ]:


template = '{0:.2f} {1:s} are worth US${2:d}'


# In[ ]:


template.format(4.5560, 'Argentine Pesos', 1)


# #### Bytes and Unicode

# In[ ]:


val = "español"
val


# In[ ]:


val_utf8 = val.encode('utf-8')
val_utf8
type(val_utf8)


# In[ ]:


val_utf8.decode('utf-8')


# In[ ]:


val.encode('latin1')
val.encode('utf-16')
val.encode('utf-16le')


# In[ ]:


bytes_val = b'this is bytes'
bytes_val
decoded = bytes_val.decode('utf8')
decoded  # this is str (Unicode) now


# #### Booleans

# In[ ]:


True and True
False or True


# #### Type casting

# In[ ]:


s = '3.14159'
fval = float(s)
type(fval)
int(fval)
bool(fval)
bool(0)


# #### None

# In[ ]:


a = None
a is None
b = 5
b is not None


# def add_and_maybe_multiply(a, b, c=None):
#     result = a + b
# 
#     if c is not None:
#         result = result * c
# 
#     return result

# In[ ]:


type(None)


# #### Dates and times

# In[ ]:


from datetime import datetime, date, time
dt = datetime(2011, 10, 29, 20, 30, 21)
dt.day
dt.minute


# In[ ]:


dt.date()
dt.time()


# In[ ]:


dt.strftime('%m/%d/%Y %H:%M')


# In[ ]:


datetime.strptime('20091031', '%Y%m%d')


# In[ ]:


dt.replace(minute=0, second=0)


# In[ ]:


dt2 = datetime(2011, 11, 15, 22, 30)
delta = dt2 - dt
delta
type(delta)


# In[ ]:


dt
dt + delta


# ### Control Flow

# #### if, elif, and else

# if x < 0:
#     print('It's negative')

# if x < 0:
#     print('It's negative')
# elif x == 0:
#     print('Equal to zero')
# elif 0 < x < 5:
#     print('Positive but smaller than 5')
# else:
#     print('Positive and larger than or equal to 5')

# In[ ]:


a = 5; b = 7
c = 8; d = 4
if a < b or c > d:
    print('Made it')


# In[ ]:


4 > 3 > 2 > 1


# #### for loops

# for value in collection:
#     # do something with value

# sequence = [1, 2, None, 4, None, 5]
# total = 0
# for value in sequence:
#     if value is None:
#         continue
#     total += value

# sequence = [1, 2, 0, 4, 6, 5, 2, 1]
# total_until_5 = 0
# for value in sequence:
#     if value == 5:
#         break
#     total_until_5 += value

# In[ ]:


for i in range(4):
    for j in range(4):
        if j > i:
            break
        print((i, j))


# for a, b, c in iterator:
#     # do something

# #### while loops

# x = 256
# total = 0
# while x > 0:
#     if total > 500:
#         break
#     total += x
#     x = x // 2

# #### pass

# if x < 0:
#     print('negative!')
# elif x == 0:
#     # TODO: put something smart here
#     pass
# else:
#     print('positive!')

# #### range

# In[ ]:


range(10)
list(range(10))


# In[ ]:


list(range(0, 20, 2))
list(range(5, 0, -1))


# seq = [1, 2, 3, 4]
# for i in range(len(seq)):
#     val = seq[i]

# sum = 0
# for i in range(100000):
#     # % is the modulo operator
#     if i % 3 == 0 or i % 5 == 0:
#         sum += i

# #### Ternary expressions

# value = 

# if 

# In[ ]:


x = 5
'Non-negative' if x >= 0 else 'Negative'

