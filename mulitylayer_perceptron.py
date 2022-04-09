import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

def actfun(a):
    if a>0:
        return 1
    else:
        return 2
def sigmod(x,der=False):
    if der==True:
        return 1/(1+ np.exp(- x))*(1-1/(1+ np.exp(- x)))
    else:
        return 1/(1+ np.exp(- x))
def ReLU(x,der=False):
    if der==True:
        return np.heaviside(x, 1)
    else:
        return np.maximum(x, 0)
def Distinguish(x):
    if x>=0.5:
        return 2
    if x<0.5:
        return 1
P="./DataSet/"
allfile=os.listdir(P)
learning_rate=input("Please set learning rate :")
n=input("Pleas set Convergence condition :")
h_n=input("Please set hidden layer neuron number :")
h_n=int(h_n)
n=int(n)
n0=n
learning_rate=float(learning_rate)
print("Please choose a file:")
print(pd.DataFrame(allfile,columns=["Data_name"]))
choose_file=input()
train_list=[]
test_list=[]
for f in allfile:
    #print(f)
    if choose_file == f:
        ff=f
        opf=open(P+f,"r")
        line=opf.readlines()
        random.shuffle(line)
        cot=1
        for l in line:
            if cot%3==0:
                test_list.append(l)
            else:
                train_list.append(l)
            cot+=1
flag=0
if f=='perceptron1.txt' or f=='perceptron2.txt':
    flag=1
random.shuffle(test_list)
random.shuffle(train_list)
te_l=[]
te_r=[]#test預期結果
tr_l=[]
tr_r=[]#train預期結果
for l in test_list:
    sp_l=l.split( )
    if flag==1:
        te_r.append(float(sp_l[-1])+1)
    else:
        te_r.append(float(sp_l[-1]))
    sp_l=sp_l[0:-1]
    ll=[-1]
    for sp in sp_l:
        ll.append(float(sp))
    te_l.append(ll)
for l in train_list:
    sp_l=l.split( )
    if flag==1:
        tr_r.append(float(sp_l[-1])+1)
    else:
        tr_r.append(float(sp_l[-1]))
    sp_l=sp_l[0:-1]
    ll=[-1]
    for sp in sp_l:
        ll.append(float(sp))
    tr_l.append(ll)
test_list=np.array(te_l)
train_list=np.array(tr_l)
"""
print("test")0
print(pd.DataFrame(test_list))
print("train")
print(pd.DataFrame(train_list))
"""
hn1=hn2=h_n#隱藏層1&2神經元個數
d1=test_list.shape[1]
w1,w2,wOut=2*np.random.rand(hn1,d1)-0.5,2*np.random.rand(hn2,hn1)-0.5,2*np.random.rand(hn2)-0.5
b1,b2,bOut=np.random.rand(hn1),np.random.rand(hn2),np.random.rand(1)
mu=[]
print(train_list.shape,w1.shape,w2.shape)
print(len(tr_r))
"""
隱藏曾共兩層
每層共4個神經元
鍵結值任意生成
"""
while n>0:
    acu=0
    for train in train_list:
        #前饋
        hn1O=ReLU(np.dot(w1,train)+b1)
        hn2O=ReLU(np.dot(w2,hn1O)+b2)
        O=sigmod(np.dot(wOut,hn2O)+bOut)
        #print(hn1O.shape,"  ",hn2O.shape," ",O," ",train.shape)
        #倒傳
        O_error=(O-tr_r[acu]+1)*sigmod(O,der=True)
        hn2_error=O_error*wOut*ReLU(hn2O,der=True)
        hn1_error=np.dot(hn2_error,w2)*ReLU(hn1O,der=True)
        #print(O_error," ",hn1_error," ",hn2_error)
        wOut=wOut-learning_rate*O_error*hn2O
        bOut=bOut-learning_rate*O_error
        w2=w2-learning_rate*np.kron(hn2_error,hn1O).reshape(hn2,hn1)
        b2=b2-hn2_error*learning_rate
        w1=w1-learning_rate*np.kron(hn1_error, train).reshape(hn1,d1)
        b1=b1-hn1_error*learning_rate
        mu.append((1/2)*(O-tr_r[acu]+1)**2)
        #print(mu[-1])
        acu+=1
        n=n-1
        if n<0:
            break

plt.scatter(np.arange(n0+1),mu,s=10)
plt.show()
plt.close()

acu=0
train_correct=0
x=[]
y=[]
C=[]
r=0
b=0
for train in train_list:
    hn1O=ReLU(np.dot(w1,train)+b1)
    hn2O=ReLU(np.dot(w2,hn1O)+b2)
    O=sigmod(np.dot(wOut,hn2O)+bOut)
    x.append(train[1])
    y.append(train[2])
    #print(O)
    if Distinguish(O)==1:
        C.append('r')
    else:
        C.append('b')
    if tr_r[acu]==Distinguish(O):
        train_correct+=1
    acu+=1
print("訓練正確率: ",train_correct/len(train_list)*100,"%")
acu=0
test_correct=0
for test in test_list:
    hn1O=ReLU(np.dot(w1,test)+b1)
    hn2O=ReLU(np.dot(w2,hn1O)+b2)
    O=sigmod(np.dot(wOut,hn2O)+bOut)
    x.append(test[1])
    y.append(test[2])
    #print(O)
    if Distinguish(O)==1:
        C.append('r')
    else:
        C.append('b')
    if te_r[acu]==Distinguish(O):
        test_correct+=1
    acu+=1
print("測試正確率: ",test_correct/len(test_list)*100,"%")
"""
#print(C)
#print('r : ',r,' b  : ',b)
a,b=float(-w[1]/w[2]),float(w[0]/w[2])
#print('a ',a,' b ',b)
X=np.linspace(min(x),max(x))
Y=a*X+b
plt.plot(X,Y,'g-')
"""
plt.scatter(x,y,c=C)
plt.title('Test Result')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.close()
##pyinstaller -F .\perceptron.py
