import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #For text features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np


# Load the dataset
df = pd.read_csv('Res_fixed.csv')


# Identify the features and the label
X = df[['N_events','Median_Time','Ratio_deletes','Ratio_pastes','Length_per_event']]  # Adjust the column names
y = df['Label'] 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Feature Extraction using TF-IDF
# tfidf_vectorizer = TfidfVectorizer(max_features=5000)
# X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
# X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# # Predict on the test set
# y_pred = rf_classifier.predict(X_test)

# Predict probabilities for the positive class
y_pred_probability = rf_classifier.predict_proba(X_test)[:, 1]


# Convert probabilities to predicted labels based on a chosen threshold
threshold = 0.5  # This threshold can be adjusted
y_pred_binary = (y_pred_probability >= threshold).astype(int)
y_pred = np.where(y_pred_binary == 1, 'Genuine', 'Fake')


# Attach these probabilities to the test set DataFrame
X_test_with_probabilities = X_test.copy()
X_test_with_probabilities['Predicted_Probability'] = y_pred_probability
X_test_with_probabilities['Actual_Label'] = y_test.reset_index(drop=True)  # Reset index to align with X_test indices

# Now you can print the DataFrame with the actual labels and predicted probabilities
print(X_test_with_probabilities[['Actual_Label', 'Predicted_Probability']])


# Print the number of samples in each set
print(f"Number of samples in training data: {len(X_train)}")
print(f"Number of samples in test data: {len(X_test)}")

# Print the number of samples for each class in the training and test sets:
print(f"Number of samples per class in training data: {dict(pd.Series(y_train).value_counts())}")
print(f"Number of samples per class in test data: {dict(pd.Series(y_test).value_counts())}")

# # Evaluate the model
# print("Confusion Matrix:")
# # print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred, zero_division=0))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))



