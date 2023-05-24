import numpy as np
import pandas as pd
import os


df = pd.read_csv("true_pred.csv")
df1 = pd.read_csv("sangebijiao.csv")

# df1.rename(columns={'predict':'GEP'})
# 删除
df.drop(columns="true", inplace=True)

# df.rename(columns={'predict':'bayes'})
df.columns=df.columns.str.replace('predict','wangge')
print(df)
df = pd.merge(df,df1,left_index=True,right_index=True)
df.to_csv("sangebijiao.csv",index=False)
