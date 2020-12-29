from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


import numpy as np
from sklearn.datasets import fetch_openml


pca = joblib.load("pca_feature_50.pkl")

# Load data from https://www.openml.org/d/554
mnist = fetch_openml('mnist_784', version=1)



features = np.array(mnist.data, 'int16')
features = pca.transform(features)
print(features.shape)
labels = np.array(mnist.target, 'int')


# digits = datasets.load_digits()

# n_samples = len(digits.images)
# print(digits['data'].shape)
# data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.1)

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.1, shuffle=False)
print("train start")
# Learn the digits on the train subset
clf.fit(X_train, y_train)
print("traind done")
# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")

joblib.dump(clf, "digits_cls_70.pkl", compress=3)

##

