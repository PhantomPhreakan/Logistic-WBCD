# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics, cross_validation
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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
log = LogisticRegression(random_state=10, solver='lbfgs')
model = log.fit(xx, yy)
#theta=np.linspace(0,1,num=100)
theta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
decisions=[]
TPR=[]
FPR=[]
decisions=model.predict_proba(xx)[:,1]
#for i in range(0,len(temp)):
#    decisions.append(temp[i,1])
for j in range(0,len(theta)):
    TP=0
    FP=0
    for i in range(0,len(xx)):
        if(decisions[i]>=theta[j]):
            yhat=4.0
        else:
            yhat=2.0
        if(yhat==4.0):
            if(yhat==yy[i]):
                TP=TP+1
            else:
                FP=FP+1
    TPR.append(TP/list(yy).count(2.0))
    FPR.append(FP/list(yy).count(4.0))
    print("TPR:",TP/len(xx),"FPR:",FP/len(xx))
TPR=[y for x,y in sorted(zip(FPR,TPR))]
FPR=sorted(FPR)
plt.plot(FPR,TPR)
plt.xlabel('FPR') 
plt.ylabel('TPR') 
plt.show()