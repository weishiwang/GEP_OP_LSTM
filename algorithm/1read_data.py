import pandas as pd
import numpy as np
import os

df = pd.read_excel("../data/黄金2016-2022数据.xls")
df['Date'] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df.set_index(keys="Date", inplace=True, drop=True)
df = df.replace({'-': np.nan})
df.dropna(inplace=True)
df = df.sort_index(ascending=True)
df.drop(columns="Adj Close**", inplace=True)
df = df.astype('float32')
data = df.values
print(len(data))


def z_score(x, mean, std):
    return (x - mean) / std


def split_data(data, width):
    x, y = [], []
    i = 0
    while (i < len(data) - width):
        x.append(data[i:i + width])
        y.append(data[i + width][3])
        i = i + 1
    return np.array(x), np.array(y)


m = np.mean(data, axis=0)
s = np.std(data, axis=0)
data = z_score(data, m, s)
np.savez("../data/mean_std.npz", mean=m[3], std=s[3])
x, y = split_data(data, 7)
# print(len(x),len(y))
# print(x[1472])
# print(y)
# np.save("../data/x.npy",x)
np.savez("../data/data_for_model.npz", x=x, y=y)
