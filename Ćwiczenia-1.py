import pandas as pd
import numpy as np

# Zadanie 3
df = pd.read_csv("../dane/_info-data-discrete.txt", sep=" ", header=None)
df1 = pd.read_csv("../dane/australian.txt", sep=" ", header=None)
df2 = pd.read_csv("../dane/australian.txt", sep=" ", header=None)

# Zadanie 3a
print("\nZadanie 3a ------------------------------------------------------------\n")

print(df2[df2.columns[-1]].unique())

# Zadanie 3b
print("\nZadanie 3b ------------------------------------------------------------\n")

print(df2[df2.columns[-1]].value_counts())

# Wczytanie plików
df1 = pd.read_csv("../dane/australian-type.txt", sep=" ", header=None)
df2 = pd.read_csv("../dane/australian.txt", sep=" ", header=None)

print(df2)

# Zadanie 3c
print("\nZadanie 3c ------------------------------------------------------------\n")

numerical = [1, 2, 6, 9, 12, 13]
print(df2[numerical].max())
print(df2[numerical].min())

# Zadanie 3d
print("\nZadanie 3d ------------------------------------------------------------\n")

df_d = pd.read_csv("../dane/australian.txt", sep=" ", header=None)
print(df_d.nunique())

# Zadanie 3e
print("\nZadanie 3e ------------------------------------------------------------\n")

df_e = pd.read_csv("../dane/australian.txt", sep=" ", header=None)
for column in df_e.columns:
    unique_values = df_e[column].unique()
    print(f"{column}: {list(unique_values)}")

# Zadanie 3f
print("\nZadanie 3f ------------------------------------------------------------\n")

df_f = pd.read_csv("../dane/australian.txt", sep=" ", header=None)
for column in df_f.columns:
    std_dev = df_f[column].std()
    print(f"{column}: {std_dev}")

one = df_f.where(df_f[14] == 1).dropna()
for column in one.columns:
    std_dev = one[column].std()
    print(f"{column}: {std_dev}")

zero = df_f.where(df_f[14] == 0).dropna()
for column in zero.columns:
    std_dev = zero[column].std()
    print(f"{column}: {std_dev}")

# Zadanie 4a
print("\nZadanie 4a ------------------------------------------------------------\n")

df_a = pd.read_csv("../dane/australian.txt", sep=" ", header=None)
total_cells = df_a.size
num_missing = int(total_cells * 0.1)

np.random.seed(42)
missing_indices = [
    (row, col)
    for row in np.random.choice(df_a.index, num_missing)
    for col in np.random.choice(df_a.columns, 1)
]

for row, col in missing_indices:
    df_a.at[row, col] = np.nan

fixed_data = df_a.copy()

for column in fixed_data.columns:
    if fixed_data[column].dtype in ['float64', 'int64']:  # Numeryczne
        mean_value = fixed_data[column].mean()
        fixed_data[column] = fixed_data[column].fillna(mean_value)
    else:  # Symboliczne
        most_common_value = fixed_data[column].mode()[0] if not fixed_data[column].mode().empty else ''
        fixed_data[column] = fixed_data[column].fillna(most_common_value)

print(fixed_data)

# Zadanie 4b
print("\nZadanie 4b ------------------------------------------------------------\n")

def normalize(i, j, a, b):
    return (((df[i][j] - min(df[i]))* (b - a)) / (max(df[i]) - min(df[i]))) + a

df_b = pd.read_csv("../dane/australian.txt", sep=" ", header=None)
print(normalize(1, 1, -10, 10))

# Zadanie 4c
print("\nZadanie 4c ------------------------------------------------------------\n")


def standardize(i, j, df):
    if i not in df.columns:
        raise KeyError(f"Column {i} not found in DataFrame")
    if j not in df.index:
        raise KeyError(f"Row {j} not found in DataFrame")

    mean = df[i].mean()
    std_dev = df[i].std()

    if std_dev == 0:
        raise ValueError("Standard deviation is zero, cannot standardize")

    df[i] = df[i].astype(float)
    df.loc[j, i] = (df.loc[j, i] - mean) / std_dev
    return df.loc[j, i]

print(standardize(1, 1, df))

# Zadanie 4d
print("\nZadanie 4d ------------------------------------------------------------\n")

df_b = pd.read_csv("../dane/Churn_Modelling.csv")
print(pd.get_dummies(df_b, columns=['Geography'], drop_first=True))
