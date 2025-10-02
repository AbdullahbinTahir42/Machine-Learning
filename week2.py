import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Create a sample dataset
# -----------------------------
data = {
    "Name": ["Ali", "Sara", "Ali", "John", "Sara"],
    "Age": [25, np.nan, 25, 40, 35],
    "Country": ["USA", "U.S.A.", "USA", "Canada", None],
    "Income": [50000, 60000, 50000, 1200000, 70000],  # has outlier
    "Purchased": ["Yes", "No", "Yes", "No", "Yes"]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

# -----------------------------
# 2. Remove duplicates
# -----------------------------
df = df.drop_duplicates()

# -----------------------------
# 3. Handle missing values
# -----------------------------
imputer = SimpleImputer(strategy="mean")
df["Age"] = imputer.fit_transform(df[["Age"]])  # fill missing with mean
df["Country"] = df["Country"].fillna("Unknown")

# -----------------------------
# 4. Standardize inconsistent values
# -----------------------------
df["Country"] = df["Country"].replace({"U.S.A.": "USA"})

# -----------------------------
# 5. Handle outliers (capping)
# -----------------------------
Q1 = df["Income"].quantile(0.25)
print(Q1)
Q3 = df["Income"].quantile(0.75)
print(Q3)
IQR = Q3 - Q1
upper_limit = Q3 + 1.5*IQR
print(upper_limit)
df["Income"] = np.where(df["Income"] > upper_limit, upper_limit, df["Income"])
lower_limit = Q1 - 1.5*IQR
print(lower_limit)
df["Income"] = np.where(df["Income"] < lower_limit, lower_limit, df["Income"])

# -----------------------------
# 6. Encode categorical variables
# -----------------------------
encoder = OneHotEncoder(sparse_output=False, drop="first")
country_encoded = encoder.fit_transform(df[["Country"]])
country_df = pd.DataFrame(country_encoded, columns=encoder.get_feature_names_out(["Country"]))

df = pd.concat([df.drop("Country", axis=1), country_df], axis=1)

# -----------------------------
# 7. Scale numeric features
# -----------------------------
scaler = StandardScaler()
df[["Age", "Income"]] = scaler.fit_transform(df[["Age", "Income"]])

print("\nCleaned & Processed Data:\n", df)
