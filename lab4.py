# #Activity 1

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE, ADASYN

# df = pd.read_csv('G:\\5th semester\\Machine Learning\\creditcard.csv')
# print(df.head())

# X = df.drop('Class' , axis = 1)
# y = df['Class']

# x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

# sampler = SMOTE(random_state=42)

# X_resampled, y_resampled = sampler.fit_resample(x_train, y_train)

# from sklearn.preprocessing import StandardScaler

# scalar = StandardScaler()

# scalar.fit(x_train)

# # Apply the transformation to the resampled training data
# X_train_scaled = scalar.transform(X_resampled)

# # Apply the transformation to the independent test data
# X_test_scaled = scalar.transform(x_test)

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, f1_score

# # --- SIMPLIFIED STEP: Train a single KNN model directly ---

# # We'll choose a common starting point: k = 5
# fixed_k = 5

# # Create the KNN model with the chosen k value
# knn_model = KNeighborsClassifier(n_neighbors=fixed_k)

# print(f"\nTraining KNN model with a fixed k={fixed_k}...")

# # Train the model on the prepared (scaled and resampled) data
# knn_model.fit(X_train_scaled, y_resampled)

# # Test the model on the independent test data
# y_pred = knn_model.predict(X_test_scaled)

# # Time to see how well it performed
# print(f"\nModel Performance on Test Data (k={fixed_k}):")
# print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

# # A detailed report is super helpful for fraud detection
# print("\nFull Classification Report:")
# print(classification_report(y_test, y_pred))

# # The model is trained and results are printed!




#Activity 2
import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Configuration ---
# IMPORTANT: CHANGE THIS PATH to where 'brain_tumor_dataset' is located
DATA_DIR = 'G:\\5th semester\\Machine Learning\\brain_tumor_dataset' 
IMG_SIZE = 64
RANDOM_SEED = 42

# Define the folders to look inside and their corresponding labels
# 'Yes' will be label 1 (Tumor), 'No' will be label 0 (No Tumor)
label_map = {'n2o': 0, 'yes': 1} 
images = []
labels = []


# --- 1. Load the dataset and preprocess all MRI images ---
print("--- Step 1: Loading and Preprocessing Images ---")

for folder_name, label_value in label_map.items():
    folder_path = os.path.join(DATA_DIR, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_name}' not found at {folder_path}. Please check your DATA_DIR path.")
        # We can continue if one is missing, but better to stop if essential data is missing
        continue
        
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        
        # Skip directories if any exist in the image folder
        if os.path.isdir(img_path):
            continue

        img = cv2.imread(img_path)
        
        if img is not None:
            # Convert to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Resize to 64x64
            resized_img = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))
            
            images.append(resized_img)
            labels.append(label_value)
            
print(f"Successfully loaded {len(images)} images.")


# --- 2. Flatten and Split Data ---
print("\n--- Step 2: Flattening and Splitting Data ---")
X = np.array(images)
y = np.array(labels)

# Flatten each image (64x64 -> 4096 features)
X_flattened = X.reshape(X.shape[0], -1)

# Split data into 80% training and 20% testing (stratify maintains the tumor/no-tumor ratio)
# X_test_original is saved for visual display later
X_train, X_test, y_train, y_test, X_train_original, X_test_original = train_test_split(
    X_flattened, y, X, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)


# --- 3. Normalize features using StandardScaler() ---
print("\n--- Step 3: Normalizing Features ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled successfully.")


# --- 4. Train an SVM ---
print("\n--- Step 4: Training SVM Model ---")
# Using a linear kernel is simple and often effective for initial image classification
svm_model = SVC(kernel='linear', random_state=RANDOM_SEED) 
svm_model.fit(X_train_scaled, y_train)
print("SVM training complete.")


# --- 5. Evaluate performance ---
y_pred = svm_model.predict(X_test_scaled)

print("\n--- Step 5: Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Tumor (0)', 'Tumor (1)']))
print("\nConfusion Matrix (Actual vs Predicted):")
print(confusion_matrix(y_test, y_pred))


# --- 6. Display a few test images with predicted and actual labels ---
print("\n--- Step 6: Displaying Sample Predictions ---")

def display_samples(X_original, y_true, y_pred, count=4):
    """Displays a few test images with their true and predicted labels."""
    indices = np.random.choice(len(X_original), count, replace=False)
    
    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(indices):
        plt.subplot(1, count, i + 1)
        img = X_original[idx] 
        plt.imshow(img, cmap='gray')
        
        true_label = "Tumor (Yes)" if y_true[idx] == 1 else "No Tumor (No)"
        pred_label = "Tumor (Yes)" if y_pred[idx] == 1 else "No Tumor (No)"
        
        title_color = 'green' if y_true[idx] == y_pred[idx] else 'red'
        plt.title(f"Actual: {true_label}\nPred: {pred_label}", color=title_color)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

display_samples(X_test_original, y_test, y_pred, count=4)