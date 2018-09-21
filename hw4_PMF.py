from __future__ import division
import numpy as np
import sys

train_data = np.genfromtxt(sys.argv[1], delimiter = ",")

lam = 2
sigma2 = 0.1
d = 5

# Implement function here
def PMF(train_data):
    users = {}
    objects = {}

    for item in train_data:
        u = int(item[0])-1
        v = int(item[1])-1
    
        if u not in users:
            users[u] = [v]
        else:
            users[u].append(v)
        
        if v in objects:
            objects[v].append(u)
        else:
            objects[v] = [u]
    #print(len(users))  
    #print (len(objects))
    #print(train_data.shape)
    
    