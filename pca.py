

import numpy as np
from sklearn.datasets import fetch_openml


# Load data from https://www.openml.org/d/554
mnist = fetch_openml('mnist_784', version=1)

##





features = np.array(mnist.data, 'int16')

print(features.shape)
labels = np.array(mnist.target, 'int')



##
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.1, shuffle=False)

##


from sklearn.decomposition import PCA

pca = PCA(n_components=60)
X_train = pca.fit_transform(X_train)
X_test

import numpy as np
from sklearn.datasets import fetch_openml


# Load data from https://www.openml.org/d/554
mnist = fetch_openml('mnist_784', version=1)

##





features = np.array(mnist.data, 'int16')

print(features.shape)
labels = np.array(mnist.target, 'int')



##
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.1, shuffle=False)




from sklearn.decomposition import PCA

pca = PCA(n_components=50)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)



explained_variance = pca.explained_variance_ratio_


print(explained_variance[:50].sum())
##

from sklearn.externals import joblib


joblib.dump(pca, "pca_feature_50.pkl", compress=3)

##

 # = pca.transform(X_test)



explained_variance = pca.explained_variance_ratio_


print(explained_variance.sum())
##

from sklearn.externals import joblib


joblib.dump(pca, "pca_feature_70.pkl", compress=3)

##

