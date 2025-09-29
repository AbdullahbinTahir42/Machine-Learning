import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

df = pd.read_csv("covid_19_data.csv")

df_pakistan = df[df['Country/Region'] == 'Pakistan']


numerical_cols = df_pakistan.select_dtypes(include=np.number).columns
for col in numerical_cols:
    median_val = df_pakistan[col].median()
    df_pakistan[col].fillna(median_val, inplace=True)

# Fill missing categorical values with a placeholder
categorical_cols = df_pakistan.select_dtypes(include='object').columns
for col in categorical_cols:
    df_pakistan[col].fillna('Unknown', inplace=True)


print("\nMissing values after handling:")
print(df_pakistan.isnull().sum())

display(df_pakistan.head())
df_pakistan['ObservationDate'] = pd.to_datetime(df_pakistan['ObservationDate'])

plt.figure(figsize=(12, 6))
plt.plot(df_pakistan['ObservationDate'], df_pakistan['Confirmed'], marker='o', linestyle='-')
plt.title('Confirmed COVID-19 Cases in Pakistan Over Time')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print(df_pakistan)