import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from pymongo import MongoClient
import os
from bson import ObjectId
import numpy as np
import joblib

class UserBehaviorClassifier:
    def __init__(self, model_version="v1.0.0", test_size=0.2, random_state=42, threshold=0.3):
        self.model_file = f'model_and_data/RF_model_{model_version}.joblib'
        self.test_size = test_size
        self.random_state = random_state
        self.threshold = threshold
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)

    def load_data_from_csv(self, csv_file):
        print("loading data from csv")
        df = pd.read_csv(f"model_and_data/{csv_file}")
        return df

    def load_data_from_mongodb(self, collection_name):
        print("loading data from mongoDB")
        client = MongoClient(os.getenv('MONGO_URI'))
        db = client['llmdetection']
        collection = db[collection_name]
        data = list(collection.find({}))
        return pd.DataFrame(data)

    def train(self, df, ignore_columns=[]):
        print("train the model")
        df.drop(columns=ignore_columns, errors='ignore', inplace=True)

        # get the column names for the features and print them
        print("Column names in order : ",list(df.columns))


        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X_train, _, y_train, _ = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        self.model.fit(X_train, y_train)

    def save_model(self):
        print("saveing the model...")
        joblib.dump(self.model, self.model_file)

    def load_model(self):
        print("loading the model...")
        self.model = joblib.load(self.model_file)

    def predict(self, segment_metrics_df):
        print("making prediction on the input")

        # [:, 1] selects the probability of the second class (usually the "positive" class in binary classification) for each instance.
        y_pred_probability = self.model.predict_proba(segment_metrics_df)[:, 1]

        y_pred_binary = (y_pred_probability >= self.threshold).astype(int)
        predictions = np.where(y_pred_binary == 1, 'Fake', 'Genuine')

        segment_metrics_df['Predictions'] = predictions
        segment_metrics_df['Predicted_Probability'] = y_pred_probability
        return segment_metrics_df


# # Example Usage:

# # Initialize the classifier
# classifier = UserBehaviorClassifier(model_version="v1.0.0")

# # For training mode:
# # Option 1: Load data from CSV
# df = classifier.load_data_from_csv('your_data.csv')
# classifier.train(df, ignore_columns=['_id', 'description'])

# # Option 2: Load data from MongoDB
# df = classifier.load_data_from_mongodb('your_collection_name')
# classifier.train(df, ignore_columns=['_id', 'other_non_feature_columns'])
# classifier.save_model()


# # For prediction mode:
# segment_metrics_df = pd.DataFrame(...)  # Replace with your segment metrics DataFrame
# classifier.load_model()
# predictions = classifier.predict(segment_metrics_df)
# print(predictions)
