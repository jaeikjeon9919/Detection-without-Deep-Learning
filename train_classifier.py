import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from time import time
from sklearn import metrics
from sklearn.externals import joblib


mnist = fetch_openml('mnist_784', version=1)

features = np.array(mnist.data, 'int16')/255.0
labels = np.array(mnist.target, 'int')

features = (features - features.mean()) / features.std()


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=4)

print("train start...")

t0 = time()
clf = SVC(kernel='poly', gamma=0.1)
clf.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))




# X_test_pca = pca.transform(X_test)

predicted = clf.predict(X_test)
# predicted = clf.predict(np.concatenate((X_test_pca, X_test),1))

print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")




# joblib.dump(pca, "pca_feature_800.pkl", compress=1)
joblib.dump(clf, "digits_cls_normalized_test.pkl", compress=1)
