import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

le=LabelEncoder()
np.random.seed(42)

students_data = {
    "ID":range(1, 101),
    "Math":np.random.randint(0, 101, 100),
    "Science":np.random.randint(0, 101, 100),
    "English":np.random.randint(0, 101, 100),
    "Grade":np.random.choice(['A', 'B', 'C', 'D', 'F'], 100)
}

df_std = pd.DataFrame(students_data)
df_std.to_csv("students.csv", index=False)

#df_std.fillna(df_std.median(), inplace=True)
df_std['Grade'] = le.fit_transform(df_std['Grade'])

df_std_totals = df_std[['Math', 'Science', 'English']].sum(axis=1)
df_std = pd.concat([df_std, df_std_totals.rename('Total')], axis=1)



plt.figure(figsize=(8, 6))
plt.hist(df_std['Math'], bins=100, edgecolor='black')
plt.xlabel('Math Scores')
plt.ylabel('Frequency')
plt.title('Distribution of Math Scores')
plt.show()


plt.boxplot(df_std['Total'], vert=False)
plt.title('Box Plot of Total Scores')
plt.show()
print(df_std.head())