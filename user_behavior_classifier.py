import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from pymongo import MongoClient
import os
from bson import ObjectId
import numpy as np
import seaborn as sns

class UserBehaviorClassifier:
    def __init__(self, model_version="v1.0.0", test_size=0.2, random_state=42):
        self.model_file = f'model_and_data/RF_model_{model_version}.joblib'
        self.model_version = model_version
        self.test_size = test_size
        self.random_state = random_state
        # self.threshold = threshold
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

        # Inspect column names and data types
        print("Column names in order : ", list(df.columns))
        print("Data types:\n", df.dtypes)

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1].astype('category')  # Ensure y is categorical

        X_train, _, y_train, _ = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        self.model.fit(X_train, y_train)

    # Evaluate the model using the available data
    def evaluate_model(self, df, ignore_columns=None):
        print("Evaluating the model...")

        # Check and handle ignore_columns
        if ignore_columns is not None and ignore_columns:
            df.drop(columns=ignore_columns, errors='ignore', inplace=True)
        else:
            print("No columns to ignore.")

        print("Columns used for evaluation: ", df.columns)

        # Separating features and target
        X = df.drop('is_cheat_segment', axis=1)
        y = df['is_cheat_segment']

        # Ensure y is categorical
        if y.dtype != 'int':
            y = y.astype('category').cat.codes

        print("Data types after processing:\n", df.dtypes)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Accuracy Score:", accuracy_score(y_test, y_pred))

        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(y_test, y_pred_proba)


    def save_model(self):
        print("saveing the model...")
        joblib.dump(self.model, self.model_file)

    def load_model(self):
        print("loading the model...")
        self.model = joblib.load(self.model_file)

    def predict(self, segment_metrics_df, threshold=0.3):
        print("making prediction on the input")

        # [:, 1] selects the probability of the second class (usually the "positive" class in binary classification) for each instance.
        y_pred_probability = self.model.predict_proba(segment_metrics_df)[:, 1]

        y_pred_binary = (y_pred_probability >= threshold).astype(int)
        predictions = np.where(y_pred_binary == 1, 'Fake', 'Genuine')

        segment_metrics_df['Predictions'] = predictions
        segment_metrics_df['Predicted_Probability'] = y_pred_probability
        return segment_metrics_df


    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        #plt.show()
        # save the plot as a file
        plt.savefig(f'model_and_data/ConfusionMatrix_{self.model_version}.png')

    def plot_roc_curve(self, y_test, y_pred_proba):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        # plt.show()
        # save the plot as a file
        plt.savefig(f'model_and_data/ROC_{self.model_version}.png')


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
