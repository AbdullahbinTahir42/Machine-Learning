import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

le = LabelEncoder()

df = pd.read_csv("activity1.csv") #the dataset is from 2021 to 2024
df['Province'] = le.fit_transform(df['Province'])
to_drop = df[df['Literacy Rate (%)'].isnull()].index
df.drop(to_drop, inplace=True)
with_median = df['Population'].median()
df['Population'].fillna(with_median, inplace=True)
print(df)     

plt.scatter(df['Population'], df['Literacy Rate (%)'])
plt.xlabel('Population (millions)')
plt.ylabel('Literacy Rate (%)')
plt.title('Population vs Literacy Rate')
plt.show()


plt.boxplot(df['Literacy Rate (%)'])
plt.title('Box plot of Literacy Rate')
plt.show()