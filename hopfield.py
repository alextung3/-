"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import seaborn as sns
sns.set_palette('hls', 10)
D="./Hopfield_dataset/"
test_data="Hopfield_dataset\Basic_Testing.txt"
train_data="Basic_Training.txt"
open_test_data=open(test_data,'r')
open_train_data=open(D+train_data,'r')
test_lines=open_test_data.readlines()
N=9*12
P=3
N_sqrt=np.sqrt(N)
w=np.zeros((9,12))
print(w.shape)
P_l=[]
r=[]
for test_l in test_lines:
    test_ll=list(test_l)
    test_ll.remove('\n')
    for i in range(len(test_ll)):
        if test_ll[i]=='1':
            test_ll[i]=1
        elif test_ll[i]==' ':
            test_ll[i]=-1
    print(test_ll)
    r.append(test_ll)
    if len(test_ll)==0:
        P_l.append(r)
        r.clear()
P_l.append(r)
np_p_l=np.array(P_l)
print(np_p_l.shape)

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('Set2')


N = 400
P = 100
N_sqrt = np.sqrt(N).astype('int32')
NO_OF_ITERATIONS = 40
NO_OF_BITS_TO_CHANGE = 200

epsilon = np.asarray([np.random.choice([1, -1], size=N)])
for i in range(P-1):
    epsilon = np.append(epsilon, [np.random.choice([1, -1], size=N)], axis=0)

print(epsilon.shape)

random_pattern = np.random.randint(P)
test_array = epsilon[random_pattern]
random_pattern_test = np.random.choice([1, -1], size=NO_OF_BITS_TO_CHANGE)
test_array[:NO_OF_BITS_TO_CHANGE] = random_pattern_test

print(random_pattern)

w = np.zeros((N, N))
h = np.zeros(N)
for i in range(N):
    for j in range(N):
        for p in range(P):
            w[i, j] += (epsilon[p, i]*epsilon[p, j]).sum()
        if i==j:
            w[i, j] = 0
w /= N
print(w)
hamming_distance = np.zeros((NO_OF_ITERATIONS, P))
for iteration in range(NO_OF_ITERATIONS):
    for _ in range(N):
        i = np.random.randint(N)
        h[i] = 0
        for j in range(N):
            h[i] += w[i, j]*test_array[j]
    test_array = np.where(h<0, -1, 1)

    for i in range(P):
        hamming_distance[iteration, i] = ((epsilon - test_array)[i]!=0).sum()

fig = plt.figure(figsize = (8, 8))
plt.plot(hamming_distance)
plt.xlabel('No of Iterations')
plt.ylabel('Hamming Distance')
plt.ylim([0, N])
plt.show()