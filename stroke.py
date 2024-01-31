import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filepath = 'Data/stroke_data.csv'
data = pd.read_csv(filepath)

data["gender"].fillna(0, inplace = True)

print(data.info())
print(data.head())
print(data.isnull().sum().any())
print(data.duplicated().sum())
print(data.describe())

data_vis = data.copy()

feature_cols = [x for x in data_vis.columns if x not in 'stroke']
plt.figure(figsize=(25,35))
# loop for subplots
for i in range(len(feature_cols)):
    plt.subplot(8,5,i+1)
    plt.title(feature_cols[i])
    plt.xticks(rotation=90)
    plt.hist(data_vis[feature_cols[i]],color = "deepskyblue")
    
plt.tight_layout()
plt.show