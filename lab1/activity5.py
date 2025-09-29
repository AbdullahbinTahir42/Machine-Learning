import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

le = LabelEncoder()

data = {
    "Name": ["Ahsan", "Hira", "Bilal", "Zara", "Salman", "Mahnoor"],
    "Age": [25, 27, 35, 29, None, 40],
    "Salary": [50000, None, 75000, 2000000, 60000, 90000],
    "Department": ["IT", "Finance", "IT", "HR", "Finance", "IT"]
}

df = pd.DataFrame(data)

age_median = df["Age"].median()
df["Age"] = df["Age"].fillna(age_median)
salary_median = df["Salary"].median()
df["Salary"] = df["Salary"].fillna(salary_median)

df["Department"] = le.fit_transform(df["Department"])

print("After handling missing values and encoding categorical data:")
print(df.head())

plt.boxplot(df["Salary"])
plt.title("Box plot of Salary")
plt.ylabel("Salary")    
plt.show()