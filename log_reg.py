#def logreg(int n,list data):
import numpy as np
import numpy as np
import pandas as pd
#def logistic(n,wc,x,y):
W=[] 
count=0

n=0.3
count=2
#x=[[2.7,2.5],[1.5,2.3],[3.4,4.4],[1.3,1.9],[3.1,3],[7.6,2.8],[5.3,2.1],[6.9,1.8],[8.7,-0.2],[7.7,3.5]]
#y=[0,0,0,0,0,1,1,1,1,1]
#W=[]

                         
miss = ['?']
xx=pd.read_csv('WBCD.csv',header=None,na_values=miss)
xx.dropna(inplace=True)
yy=xx[10]
data=[]
xx=xx[[1,2,3,4,5,6,7,8,9]]



xx=xx.values.tolist()
yy=yy.values.tolist()
xx=np.array(xx,dtype=float)
yy=np.array(yy,dtype=float)
#logistic(n,10,xx,yy) 
wc=10
x=xx
y=yy

for i in range (0,wc):
        W.append(0)  

X=[]
while(count<=2):
    for j in range(0,len(x)):
        X=x[j]
        X= [1, *X]
        P=1/(1+np.exp(-np.dot(W,X)))
        if(P>=0.5):
            yhat=2.0
        else:
            yhat=4.0
        if(yhat!=y[j]):
#            count+=1
            for i in range(0,wc):
#                   sigmoid=1/(1+np.exp(-np.dot(W,X)))
                   W[i]=W[i]+n*X[i]*(y[j]-P)
#        else:
#            count+=1
    print("Epoch :",W)
    count+=1