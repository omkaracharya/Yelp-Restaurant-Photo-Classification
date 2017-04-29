import numpy as np
import pandas as pd
import time
import os

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.externals import joblib


def get_labels(label_string):
    """
        This function converts label from string to array of labels
        Input: "(1, 2, 3, 4, 5)"
        Output: [1, 2, 3, 4, 5]
    """
    label_array = label_string[1:-1]
    label_array = label_array.split(',')
    label_array = [int(label) for label in label_array if len(label) > 0]
    return label_array


def get_features(feature_string):
    """
        This function converts feature vector from string to array of features
        Input: "(1.2, 3.4, ..., 9.10)"
        Output: [1.2, 3.4, ..., 9.10]
    """
    feature_array = feature_string[1:-1]
    feature_array = feature_array.split(',')
    feature_array = [float(label) for label in feature_array]
    return feature_array


# Set home paths for data and features
DATA_HOME = '../data/'
FEATURES_HOME = '../features/'
MODELS_HOME = '../models/'

print "Support Vector Machine on Train and Validation sets"

# Read training data and test data
train_data = pd.read_csv(FEATURES_HOME + 'train_business_features.csv')

# Separate the labels from features in the training data
trainX = np.array([get_features(feature) for feature in train_data['feature']])
trainY = np.array([get_labels(label) for label in train_data['label']])

# Use validation data for calculating the training accuracy, random_state ensures reproducible results without overfitting
trainX, validationX, trainY, validationY = train_test_split(trainX, trainY, test_size=0.3, random_state=42)

# Binary representation (just like one-hot vector) (1, 3, 5, 9) -> (1, 0, 1, 0, 1, 0, 0, 0, 1)
mlb = MultiLabelBinarizer()
trainY = mlb.fit_transform(trainY)

# Do the same for validation labels
actual_labels = validationY
mlb = MultiLabelBinarizer()
validationY = mlb.fit_transform(validationY)

if not os.path.isfile(MODELS_HOME + 'svm_model_for_validation.pkl'):
    print "Model training started."

    # Start time
    start_time = time.time()

    # Create an SVM classifier from sklearn package
    clf = OneVsRestClassifier(SVC(kernel='linear', probability=True, verbose=True))

    # Fit the classifier on the training data and labels
    clf.fit(trainX, trainY)

    print "Model trained."

    joblib.dump(clf, MODELS_HOME + 'svm_model_for_validation.pkl')
    print "Model saved."

    # End time
    end_time = time.time()

    print "Time taken for training the SVM model:", end_time - start_time, "sec"

clf = joblib.load(MODELS_HOME + 'svm_model_for_validation.pkl')

print "Model loaded."

# Predict the labels for the validation data
svm_preds_binary = clf.predict(validationX)

# Predicted labels are converted back
# (1, 0, 1, 0, 1, 0, 0, 0, 1) -> (1, 3, 5, 9)
predicted_labels = mlb.inverse_transform(svm_preds_binary)

print "Validation Set Results:"
print "Overall F1 Score:", f1_score(svm_preds_binary, validationY, average='micro')
print "Individual F1 Score:", f1_score(svm_preds_binary, validationY, average=None)
