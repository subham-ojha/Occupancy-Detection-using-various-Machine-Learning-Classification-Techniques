
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier

import time

def squared_error(actual, pred):
    return (pred - actual) ** 2
def check(actual, pred):

    if np.all(actual==pred):
        return 1
    else:
        return 0

train=pd.read_csv("datatraining.csv",header=None)
test_opendoor=pd.read_csv("datatest.csv",header=None)
test_closeddoor=pd.read_csv("datatest2.csv",header=None)

X_train = train.iloc[: , 2:6]
Y_train = train.iloc[: , 7]
X_test = test_opendoor.iloc[: , 2:6]
Y_test = test_opendoor.iloc[: , 7]
X1_test = test_closeddoor.iloc[: , 2:6]
Y1_test = test_closeddoor.iloc[: , 7]


def model(model):
    return model()
l=[LinearSVC, KNeighborsClassifier,ensemble.RandomForestClassifier, tree.DecisionTreeClassifier]
for t in l:
    t1=time.time()
    clf = MultiOutputClassifier(l, n_jobs=-1)

    clf=model(t)
    clf = clf.fit(X_train, Y_train)
    time_taken = time.time() - t1
    predicted_opendoor=clf.predict(X_test)
    predicted_closeddoor=clf.predict(X1_test)
    error=0
    correct=0
    for i in range(len(X_test)):
        error+=squared_error(Y_test[i],predicted_opendoor[i])
        correct+=check(Y_test[i],predicted_opendoor[i])
        
    for i in range(len(X1_test)):
        error+=squared_error(Y1_test[i],predicted_closeddoor[i])
        correct+=check(Y1_test[i],predicted_closeddoor[i])
        
    Mse_opendoor=error/len(X_test)
    Mse_closeddoor=error/len(X1_test)
    
    conf_mat_opendoor=confusion_matrix(Y_test, predicted_opendoor)
    conf_mat_closeddoor=confusion_matrix(Y1_test, predicted_closeddoor)
    
    print("********************************************")
    print("*****PARAMETERS IN RESPECT OF OPEN DOOR*****")
    print("********************************************")
    
    print("Time taken for {} is {}".format(t, time_taken))
    print(clf)
    print("For Model {} mean squared Error is {}".format(t,Mse_opendoor))
    print("For Model {} Accuracy is {} percent".format(t,accuracy_score(Y_test, predicted_opendoor)))
    print("For Model {} F_Score is {} percent".format(t,f1_score(Y_test, predicted_opendoor, average="macro")))
    print("For Model {} Precision is {} percent".format(t,precision_score(Y_test, predicted_opendoor, average="macro")))
    print("For Model {} Recall is {} percent".format(t,recall_score(Y_test, predicted_opendoor, average="macro")))

    print("********************************************")
    print("*****PARAMETERS IN RESPECT OF CLOSED DOOR*****")
    print("********************************************")
    
    print("Time taken for {} is {}".format(t, time_taken))
    print(clf)
    print("For Model {} mean squared Error is {}".format(t,Mse_closeddoor))
    print("For Model {} Accuracy is {} percent".format(t,accuracy_score(Y1_test, predicted_closeddoor)))
    print("For Model {} F_Score is {} percent".format(t,f1_score(Y1_test, predicted_closeddoor, average="macro")))
    print("For Model {} Precision is {} percent".format(t,precision_score(Y1_test, predicted_closeddoor, average="macro")))
    print("For Model {} Recall is {} percent".format(t,recall_score(Y1_test, predicted_closeddoor, average="macro")))


