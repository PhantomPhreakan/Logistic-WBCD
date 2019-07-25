import numpy as np
import math as m

theata = 0.5
n = 0.3
w = [0,0,0]

x = [[2.7,2.5],
     [1.5,2.3],
     [3.4,4.4],
     [1.3,1.9],
     [3.1,3],
     [7.6,2.8],
     [5.3,2.1],
     [6.9,1.8],
     [8.7,-0.2],
     [7.7,3.5]]
y = [0,0,0,0,0,1,1,1,1,1]

def f(w,x):
    return 1/(1+m.exp(-np.dot(w,[1]+x)))

def check(w,x,y,i):
    ret=''
    pred = f(w,x[i])
    print(pred)
    if(pred >= theata):
        if(y[i]==1):
            ret='tp'
        else:
            ret='fp'
    else:
        if(y[i]==1):
            ret='fn'
        else:
            ret='tn'
    return ret

def stoch(x,y,w,theata,n,no):  
    count = 0
    ep_no = 0
    i = 0
    pred = 0
    while(count < len(y)-1):
        print(w)
        if(i==no):
            i+=1
        i = i%len(y)
        if(i%len(y)==0):
            ep_no+=1
        ypred = f(w,x[i])
        if(ypred >= theata):
            pred = 1
        else:
            pred = 0
        if(pred == y[i]):
            count+=1
        else:
            w = w+np.dot(n*(y[i]-ypred),[1]+x[i])
            count = 0
        i+=1
    return w

v = {'tp':0,'tn':0,'fp':0,'fn':0}

for i in range(len(x)):
    print('i')
    wtemp = stoch(x,y,w,theata,n,i)
    v[check(wtemp,x,y,i)]+=1

print('Accuracy: ',((v['tp']+v['tn'])/len(x)))
print('Postive precision: ',(v['tp']/(v['tp']+v['fp'])))
print('Negative precision: ',(v['tn']/(v['tn']+v['fn'])))
print('Positive recall: ',(v['tp']/(v['tp']+v['fn'])))
print('Negative recall: ',(v['tn']/(v['tn']+v['fp'])))
    
    