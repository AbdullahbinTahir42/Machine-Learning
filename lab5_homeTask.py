import pandas as pd
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# Step 1: Load the dataset
mushroom_data = pd.read_csv("mushrooms.csv")

# Display first few rows
print("Dataset Preview:")
print(mushroom_data.head())

# Step 2: Check for missing values
print("\nMissing Values in Each Column:")
print(mushroom_data.isnull().sum())

print(f"Dataset shape: {mushroom_data.shape}")
print(f"Target distribution:\n{mushroom_data['class'].value_counts()}")

# plt.figure(figsize=(15, 10))
# plt.subplot(2, 3, 1)
# sns.countplot(data=mushroom_data, x='class')
# plt.title('Target Distribution')
# plt.subplot(2, 3, 2)
# sns.countplot(data=mushroom_data, x='cap-shape', hue='class')
# plt.title('Cap Shape vs Class')
# plt.subplot(2, 3, 3)
# sns.countplot(data=mushroom_data, x='cap-color', hue='class')
# plt.title('Cap Color vs Class')

# plt.subplot(2, 3, 4)

# sns.countplot(data=mushroom_data, x='population')

# plt.title('Population Distribution')
# plt.subplot(2, 3, 5)

# sns.countplot(data=mushroom_data, x='habitat')

# plt.title('Habitat Distribution')
# plt.subplot(2, 3, 6)

# sns.countplot(data=mushroom_data, x='odor', hue='class')

# plt.title('Odor vs Class')

# plt.tight_layout()
# plt.show()



mushroom_df = mushroom_data.dropna()
# Separate features and target
mushroom_features = mushroom_df.drop('class', axis=1)
mushroom_labels = mushroom_df['class']
# Encode categorical features
encoder = OrdinalEncoder()
mushroom_prepared = encoder.fit_transform(mushroom_features)
mushroom_prep_df = pd.DataFrame(mushroom_prepared, columns=mushroom_features.columns)

k_clust = KMeans(n_clusters=2, random_state=42)

k_clust.fit(mushroom_prep_df)

# Get results

k_labels = k_clust.labels_

map_dict = {'e': 0, 'p': 1}
true_labels = mushroom_labels.map(map_dict)

# Calculate accuracy
accuracy = accuracy_score(true_labels, k_labels)
print(f"KMeans Accuracy: {accuracy:.4f}")


plt.figure(figsize=(8, 6))

cm = confusion_matrix(true_labels, k_labels)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',  xticklabels=['Cluster 0', 'Cluster 1'], yticklabels=['Edible', 'Poisonous'])

plt.title('Confusion Matrix')
plt.xlabel('Predicted Clusters')
plt.ylabel('True Labels')
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(true_labels, k_labels, target_names=['Edible', 'Poisonous']))
