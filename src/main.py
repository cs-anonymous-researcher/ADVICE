#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %%

import numpy as np
import pandas as pd
from os.path import join as p_join
from matplotlib import pyplot as plt
import json, os, time, re, math
import pickle

# %%

data_path = "../../DBGenerator/imdb-benchmark/"      # 
table_name = "cast_info.csv"                         #
header_names = ["id", "person_id", "movie_id", "person_role_id", "note", "nr_order", "role_id"]

data_df = pd.read_csv(p_join(data_path, table_name), names=header_names, \
    header = None, quotechar = '"', escapechar = "\\")
data_df.head()

# %%
val1 = data_df['person_id'].values
print(val1.shape)
# %%
val2 = data_df['movie_id'].values
print(val2.shape)

# %%
pd.unique(data_df['role_id'])
# %%
len(sorted(pd.unique(data_df['person_role_id'].dropna())))
# %%
len(data_df)
# %%
res = data_df.groupby(["person_role_id"]).count()
# %%
person_role_id_list = res['id'].values
person_role_id_list.sort()
person_role_id_list = person_role_id_list[::-1]
person_role_id_list
# %%
plt.plot(person_role_id_list)
plt.yscale("log")
# %%

res = data_df.groupby(["movie_id"]).count()
movie_id_list = res['id'].values
movie_id_list.sort()
movie_id_list = movie_id_list[::-1]

# %%

plt.plot(movie_id_list)
# %%
from scipy.stats import skew

skew(movie_id_list)

# %%

import rdc
import numpy as np

a = np.array([6,5,4,3,2])
b = np.array([2,3,4,5,6])

rdc.rdc(a, b)
rdc.rdc(val1, val2)
# %%

data_path = "../../DBGenerator/imdb-benchmark/"      # 
table_name = "cast_info.csv"                         #
header_names = ["id", "person_id", "movie_id", "person_role_id", "note", "nr_order", "role_id"]

# %%

res1 = np.logical_and(val1 <= 53000, val1 >= 0)
sum1 = res1.sum()
print("sum1 = {}".format(sum1))
res2 = np.logical_and(val2 <= 60000, val1 >= 50000)
sum2 = res2.sum()
print("sum2 = {}".format(sum2))

conjunction_res = np.logical_and(res1, res2)
conjunction_sum = conjunction_res.sum()

print((sum1 * sum2 / 36244344), conjunction_sum)


# %%
(val1 <= 20000).sum()
# %%
len(val1)
# %%
272887 * 156593/ 36244344
# %%
np.max(val2)
# %%
