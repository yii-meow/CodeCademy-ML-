import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import re

from sklearn import preprocessing

df = pd.read_csv("profiles.csv")
print(df.columns)

print(df.head())

# Mapping str values to numerical
df["drinks_code"] = df.drinks.map({
    "not at all" : 0,
    "rarely" : 1,
    "socially" : 2,
    "often" : 3,
    "very often" : 4,
    "desperately" : 5
})

df["smokes_code"] = df.smokes.map({
    "no":0,
    "sometimes":1,
    "when drinking":2,
    "yes":3,
    "trying to quit":4
})

df["drugs_code"] = df.drugs.map({
    "never":0,
    "sometimes":1,
    "often":2
})

# Retrieve the essays cols
essays_cols = ["essay0","essay1","essay2","essay3","essay4","essay5",
               "essay6","essay7","essay8","essay9"]

# Removing the NaNs
all_essays = df[essays_cols].replace(np.nan,'',regex=True)

# Combining the essays
all_essays = all_essays.apply(lambda x:' '.join(x),axis=1)

# Total length of the ssays
df["essay_len"] = all_essays.apply(lambda x : len(x))

# Average length for essays
df["avg_word_length"] = df["essay_len"]/len(essays_cols)

# Frequency of the words 'I' or 'me"
df["common_words"] = all_essays.str.count(r'\b(I|me)\b', flags=re.IGNORECASE)

feature_data = df[["smokes_code","drinks_code","drugs_code",
                   "essay_len","avg_word_length"]]

x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled,columns=feature_data.columns)

# Predict Sex with education level and income
df["sex"] = df.sex.map({
    "m" : 0,
    "f" : 1
})

print(df["education"].head())