import pandas as pd

df = pd.read_csv("subject-info.csv")
df.fillna(inplace=True)

df.to_csv("subject_clean.csv")
