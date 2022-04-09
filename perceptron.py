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
P="./NN_HW1_DataSet/"
allfile=os.listdir(P)
learning_rate=input("Please set learning rate :")
n=input("Pleas set Convergence condition :")
n=int(n)
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

random.shuffle(test_list)
random.shuffle(train_list)
#print("test")
#print(pd.DataFrame(test_list))
#print("train")
#print(pd.DataFrame(train_list))
r=random.randint(0,len(train_list)-1)
sp_d=train_list[r].split( )
w=np.array([-1.0,float(sp_d[0]),float(sp_d[1])])
print(w)
while True:
    for l in train_list:
        sp_l=l.split( )
        ll=[-1,float(sp_l[0]),float(sp_l[1])]
        output=0
        if ff=='perceptron1.txt' or sp_l[2]=='perceptron2.txt':
            output=int(sp_l[2])+1
        else:
            output=int(sp_l[2])
        npll=np.array(ll)
        #print(npll,type(npll),output)
        wT=w.transpose()
        Dot_product=np.dot(wT,npll)
        af=actfun(Dot_product)
        if af==1 and output==2:
            w=w-(float(learning_rate)*npll)
        elif af==2 and output==1:
            w=w+(float(learning_rate)*npll)
        n=n-1
        if n==0:
            break
    if n==0:
        break
print('鍵結值(w): ',w)
cal=0
for l in train_list:
    sp_l=l.split( )
    ll=np.array([-1,float(sp_l[0]),float(sp_l[1])])
    wT=w.transpose()
    Dot_product=np.dot(wT,ll)
    af=actfun(Dot_product)
    if ff=='perceptron1.txt' or ff=='perceptron2.txt':
        if af==int(sp_l[2])+1:
            cal+=1
    else:
        if af==int(sp_l[2]):
            cal+=1
ans=float(cal)/float(len(train_list))*100
#print('wow')####
print('訓練資料正確率: ',ans,'%')
cal=0
for l in test_list:
    sp_l=l.split( )
    ll=np.array([-1,float(sp_l[0]),float(sp_l[1])])
    wT=w.transpose()
    Dot_product=np.dot(wT,ll)
    af=actfun(Dot_product)
    if ff=='perceptron1.txt' or ff=='perceptron2.txt':
        if af==int(sp_l[2])+1:
            cal+=1
    else:
        if af==int(sp_l[2]):
            cal+=1
ans=float(cal)/float(len(test_list))*100
print('測試資料正確率: ',ans,'%')
dic={}
x=[]
y=[]
C=[]
r=0
b=0
for l in test_list:
    sp_l=l.split( )
    sl=int(sp_l[2])
    if ff=='perceptron1.txt' or ff=='perceptron2.txt':
         sl+=1
    if sl==1:
        x.append(float(sp_l[0]))
        y.append(float(sp_l[1]))
        C.append('r')
        r+=1
    else:
        x.append(float(sp_l[0]))
        y.append(float(sp_l[1]))
        C.append('b')
        b+=1
#print(C)
#print('r : ',r,' b  : ',b)
a,b=float(-w[1]/w[2]),float(w[0]/w[2])
#print('a ',a,' b ',b)
X=np.linspace(min(x),max(x))
Y=a*X+b
plt.plot(X,Y,'g-')
plt.scatter(x,y,c=C)
plt.title('Test Result')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
plt.close()
##pyinstaller -F .\perceptron.py