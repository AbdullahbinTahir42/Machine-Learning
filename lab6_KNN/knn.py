import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import distances
# Dataset
data = {
 'Age': [25, 35, 45, 20, 50, 40, 23, 48, 33, 52],
 'Income': [40, 50, 65, 30, 70, 60, 35, 63, 48, 80],
 'Buy': ['No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
}


df = pd.DataFrame(data)


le = LabelEncoder()

df['Buy'] = le.fit_transform(df['Buy'])

X = df[['Age', 'Income']]

y = df['Buy']

plt.scatter(df['Age'],df['Buy'], color='blue')
plt.plot(df['Age'],df['Buy'], color='Black', linestyle='--', marker='o')
plt.xlabel('Age')
plt.ylabel('Buy')
plt.title('Age vs Buy')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn  = KNeighborsClassifier(n_neighbors=4, metric=distances.euclidean_distance)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
new_test = np.array([[30, 55], [45, 70]])
predictions = knn.predict(new_test)
print("Predictions for new data points:", le.inverse_transform(predictions)) 

accuracy = accuracy_score(y_test, y_pred)
print(f'Euclidean Accuracy: {accuracy * 100:.2f}%')

knn_manhattan = KNeighborsClassifier(n_neighbors=4, metric=distances.manhathan_distance)
knn_manhattan.fit(X_train, y_train) 
y_pred_manhattan = knn_manhattan.predict(X_test)
accuracy_manhattan = accuracy_score(y_test, y_pred_manhattan)   
print(f'Manhattan Accuracy: {accuracy_manhattan * 100:.2f}%')

knn_minkowski = KNeighborsClassifier(n_neighbors=4, metric=lambda a, b: distances.Minkowski_distance(a, b, 3))
knn_minkowski.fit(X_train, y_train)
y_pred_minkowski = knn_minkowski.predict(X_test)
accuracy_minkowski = accuracy_score(y_test, y_pred_minkowski)
print(f'Minkowski Accuracy: {accuracy_minkowski * 100:.2f}%')


print("Using Different k values:")
k_values = range(1, 15)
