import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import joblib

class UserBehaviorClassifier:
    def __init__(self, csv_file, model_file='model_and_data/random_forest_model_v2.joblib', test_size=0.2, random_state=42, threshold=0.5):
        self.csv_file = "model_and_data/" + csv_file
        self.model_file = model_file
        self.test_size = test_size
        self.random_state = random_state
        self.threshold = threshold
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self._load_data()

    def _load_data(self):
        # Load the dataset
        df = pd.read_csv(self.csv_file, usecols=lambda column: column != 'ID')

        # Identify the features and the label
        X = df.iloc[:, 1:-1]
        y = df['Label']

        # Split the dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # Reset indices
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        # Predict probabilities for the positive class
        y_pred_probability = self.model.predict_proba(self.X_test)[:, 1]

        # Convert probabilities to predicted labels
        y_pred_binary = (y_pred_probability >= self.threshold).astype(int)
        y_pred = np.where(y_pred_binary == 1, 'Genuine', 'Fake')

        # Attach probabilities to the test set DataFrame
        X_test_with_probabilities = self.X_test.copy()
        X_test_with_probabilities['Predicted_Probability'] = y_pred_probability
        X_test_with_probabilities['Actual_Label'] = self.y_test

        return X_test_with_probabilities

    def evaluate(self):
        y_pred_probability = self.model.predict_proba(self.X_test)[:, 1]
        y_pred_binary = (y_pred_probability >= self.threshold).astype(int)
        y_pred = np.where(y_pred_binary == 1, 'Genuine', 'Fake')

        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, zero_division=0))

        print("\nAccuracy Score:")
        print(accuracy_score(self.y_test, y_pred))

    def save_model(self):
        joblib.dump(self.model, self.model_file)

        
# Example usage:
# classifier = UserBehaviorClassifier('Synth.csv')
# classifier.train()
# predictions = classifier.predict()
# print(predictions[['Actual_Label', 'Predicted_Probability']])
# classifier.evaluate()
# classifier.save_model()
