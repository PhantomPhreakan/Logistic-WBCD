# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 00:04:04 2019

@author: 17pd03
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics, cross_validation
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
miss = ['?']
xx=pd.read_csv('WBCD.csv',header=None,na_values=miss)
xx.dropna(inplace=True)
yy=xx[10]
xx=xx[[1,2,3,4,5,6,7,8,9]]
xx=xx.values.tolist()
yy=yy.values.tolist()
xx=np.asarray(xx)
yy=np.asarray(yy)
yy=np.array(yy,dtype=float)
logreg=LogisticRegression()
logreg.fit(xx,yy)
theta=np.linspace(0,1,num=20)
decisions=[]
mythreshold=np.linspace(0,1,num=20)
for i in range(0,20):
    decisions.append((logreg.predict_proba(xx) ))
TPR=[]
kfold = KFold(10, True, 1)
for train_index, test_index in kfold.split(xx):
   # print("Train:",train,"Test:", test)
    X_train,X_test=xx[train_index],xx[test_index]
    Y_train,Y_test=yy[train_index],yy[test_index]
    #X_train, X_test, y_train, y_test = train_split(df, y, test_size=0.2)
    model=logreg.fit(X_train,Y_train)
    predictions = model.predict(X_test)
    mat=confusion_matrix(Y_test,predictions)
    print(confusion_matrix(Y_test,predictions))
    tn,fp,fn,tp=confusion_matrix(Y_test,predictions).ravel()
    print("Accuracy:",(tp+tn)/(tp+fp+tn+fn))
    print("Precision:",tp/(tp+fp))
    print("Recall:",tp/(tp+fn))
#    fpr, tpr=roc_curve(predictions,Y_test,drop_intermediate=False)

    



