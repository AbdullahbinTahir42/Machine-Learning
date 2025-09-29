import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_st = pd.read_csv("G:\\5th semester\\Machine Learning\\lab1\\Data sets\\students.csv")
df_att = pd.read_json("G:\\5th semester\\Machine Learning\\lab1\\Data sets\\attendance.json")
df_extra = pd.read_excel("G:\\5th semester\\Machine Learning\\lab1\\Data sets\\extra.xlsx")

df = pd.merge(df_st, df_att, on="Name")
df = pd.merge(df, df_extra, on="Name")

colors = np.where(df['Attendance'] < 70, 'red', 'blue')

# Use the 'c' argument to apply the colors
plt.scatter(df['Attendance'], df['Marks'], c=colors)
plt.title('Attendance vs Marks')
plt.xlabel('Attendance')
plt.ylabel('Marks')
plt.grid(True)
plt.show()


df  = pd.get_dummies(df, columns=['Name'], drop_first=True)
print(df.head())