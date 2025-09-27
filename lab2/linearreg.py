import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

lr = LinearRegression()

df = pd.read_csv('G:\\5th semester\\Machine Learning\\lab2\\DataSet\\Admission_Predict.csv')

columns = df.columns.tolist()

df.drop('Serial No.', axis=1, inplace=True)
y = df['Chance of Admit ']
X = df.drop('Chance of Admit ', axis=1,inplace=True )

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)
rmse = root_mean_squared_error(y_test, predictions)
print("Root Mean Squared Error:", rmse)


