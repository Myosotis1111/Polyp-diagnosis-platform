import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import joblib
import re

# 1. Load training and testing data from Excel files
train_file_path = '../feature_data_for_cls/training_features.xlsx'
test_file_path = '../feature_data_for_cls/test_features.xlsx'
train_data = pd.read_excel(train_file_path)
test_data = pd.read_excel(test_file_path)

# Remove rows with missing values except for 'Classification type' and 'file_name' columns
train_data.dropna(subset=train_data.columns.difference(['Classification type', 'file_name']), inplace=True)
test_data.dropna(subset=test_data.columns.difference(['Classification type', 'file_name']), inplace=True)

# Control feature selection, excluding non-feature columns
selected_features = [col for col in train_data.columns if col not in ['Classification type', 'file_name']]

# Extract target categories based on the file_name column
def extract_category(file_name):
    match = re.search(r'^[^-]*-[^-]*-(\d+)', file_name)
    if match:
        category_number = int(match.group(1))
        if category_number == 1:
            return "Hyperplastic"
        elif category_number == 2:
            return "Adenomatous"
        else:
            print(f"Unrecognized category number {category_number} in filename {file_name}")
            return None
    else:
        print(f"No match found in filename {file_name}")
        return None

# Add 'Category' column based on the file_name
train_data['Category'] = train_data['file_name'].apply(extract_category)
test_data['Category'] = test_data['file_name'].apply(extract_category)

# Filter out rows with unrecognized categories
train_data = train_data[train_data['Category'].notna()]
test_data = test_data[test_data['Category'].notna()]

# Split data into features (X) and labels (y)
X_train = train_data[selected_features]
y_train = train_data['Category']
X_test = test_data[selected_features]
y_test = test_data['Category']

# Calculate mean values of each feature for both categories in the training data
mean_values = train_data.groupby('Category')[selected_features].mean()

# Save feature means to an Excel file
mean_values_file = '../model_weight_path/SVM+sfbs/feature_means.xlsx'
mean_values.to_excel(mean_values_file)
print(f"Mean values saved as {mean_values_file}")

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler object for future use
scaler_file = '../model_weight_path/SVM+sfbs/scaler.pkl'
joblib.dump(scaler, scaler_file)
print(f"Scaler saved as {scaler_file}")

# Initialize SVM classifier with a linear kernel
model = SVC(probability=True, kernel="linear", random_state=42)

# Initialize Sequential Floating Backward Selection (SFBS) feature selector
sfs = SFS(model,
          k_features='best',  # Automatically selects the optimal number of features
          forward=False,      # Perform backward selection
          floating=True,      # Allow adding or removing features dynamically
          scoring='accuracy', # Use accuracy as the metric
          cv=5,               # 5-fold cross-validation
          n_jobs=1)           # Run in a single job

# Fit the feature selector on the scaled training data
sfs.fit(X_train_scaled, y_train)

# Retrieve the selected features
selected_features = list(sfs.k_feature_names_)

# Transform the datasets to only include the selected features
X_train_sfs = sfs.transform(X_train_scaled)
X_test_sfs = sfs.transform(X_test_scaled)

# Retrain the SVM model using the selected features
model.fit(X_train_sfs, y_train)

# Save the trained SVM model to a file
model_file = '../model_weight_path/SVM+sfbs/weight.pkl'
joblib.dump(model, model_file)
print(f"Model saved as {model_file}")

# Save the SFS selector object
sfs_file = '../model_weight_path/SVM+sfbs/selector.pkl'
joblib.dump(sfs, sfs_file)
print(f"SFS Selector saved as {sfs_file}")

# Make predictions on the test set
y_pred = model.predict(X_test_sfs)

# Get predicted probabilities
probabilities = model.predict_proba(X_test_sfs)

# Print the classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

# Print the predictions and probabilities for each sample
print("\nPredictions and Probabilities:")
for i, (true_label, pred, prob) in enumerate(zip(y_test, y_pred, probabilities)):
    print(f"Sample {i + 1}: True Class = {true_label}, Predicted Class = {pred}, Probabilities = {prob}")

# Print indices of incorrect predictions
print("\nIncorrect Predictions:")
incorrect_indices = np.where(y_pred != y_test)[0]
for i in incorrect_indices:
    print(f"Sample {i + 1}: True Class = {y_test.iloc[i]}, Predicted Class = {y_pred[i]}, Probabilities = {probabilities[i]}")

# Output the selected features
print("\nSelected Features:")
print(selected_features)

# Output discarded features
discarded_feature_names = [col for col in train_data.columns if col not in selected_features]
print("\nDiscarded Features:")
print(discarded_feature_names)

# Visualize the distribution of the selected, scaled features
plt.figure(figsize=(12, 10))
sns.boxplot(data=pd.DataFrame(X_train_sfs, columns=selected_features))
plt.title('Scaled Feature Distribution (Training Data)')
plt.xticks(rotation=90)
plt.show()

# Calculate and print feature importance based on the absolute values of the SVM coefficients
feature_importances = np.abs(model.coef_).flatten()
feature_importance_dict = dict(zip(selected_features, feature_importances))
sorted_feature_importances = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("\nFeature Importances (Descending):")
for feature, importance in sorted_feature_importances:
    print(f"Feature: {feature}, Importance: {importance}")
