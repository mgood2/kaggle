import numpy as np
import pandas as pd
from sklearn import svm, metrics


X_test = pd.read_csv('test.csv').as_matrix()
y = np.asarray(pd.read_csv('train.csv', usecols=[0]).as_matrix()).flatten()
X = pd.read_csv('train.csv', usecols=range(1,784)).as_matrix()

print y
print "\n-------------------\n"
print X
"""
n_samples = len(X)

clf = svm.SVC(gamma=0.001)
clf.fit(X[:n_samples /2], y[:n_samples /2])

expected = y[n_samples/2:]
predicted = clf.predict(X[n_samples/2:])

print("Classification report for classifier %s:\n%s\n"
        %(clf, metrics.classification_report(expected, predicted)))
"""
