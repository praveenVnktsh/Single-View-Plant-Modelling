import glob
from typing import List
import cv2
import numpy as np
from datetime import datetime
from functions import *
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import random
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    # "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]



classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]



def fitSVM():

    print('Loading data')
    leafBuffers = torch.load('leafFeatures.pt')
    stemBuffers = torch.load('stemFeatures.pt')
    random.shuffle(leafBuffers)

    leafBuffers = leafBuffers[:int(len(stemBuffers))]

    data = stemBuffers + leafBuffers
    random.shuffle(data)
    
    dataLen = 30000
    data = data[:dataLen]
    x = np.array([np.array(d['feature']) for d in data])
    y = np.array([d['label'] for d in data])

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1,)

    clf = svm.SVC(kernel='rbf') 
    

    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)

        print(name, ':',  accuracy)


        torch.save(clf, f'binary/{name}_{int(round(accuracy,2)*100)}_{dataLen}.pt')
    return clf

pred = fitSVM()

