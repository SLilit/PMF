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
    
    lam = 2
    sigma2 = 0.1
    d = 5
    I = np.identity(5)
    len_U = len(users)
    len_V = len(objects)
    M = np.zeros(shape = (len_U, len_V))        
    U = np.zeros(shape = (50,len_U,5))  
    V = np.zeros(shape = (50,len_V,5))
    L = np.zeros(shape = (50,1))
    
    for item in train_data:
        M[int(item[0]-1)][int(item[1])-1] = item[2]
    
    #v_init = np.zeros(shape = (len_V,d))
    for i in range(len(objects)):
        V[0][i] = np.dot(np.random.normal(0,1/lam,d),I)
    
    for k in range(50):
        for i in range(len_U):
            v_sum = 0
            m_sum = 0
            for j in objects[i]:
                v_sum += np.dot(V[k][j],V[k][j].reshape((-1,1)))
                m_sum += M[i][j]*V[k][j]
        U[k][i] = np.dot(np.linalg.inv(lam*sigma2*I + v_sum),m_sum)
        
        for j in range(len_V):
            u_sum = 0
            m_sum = 0
            for i in objects[j]:
                u_sum += np.dot(U[k][i],U[k][i].reshape((-1,1)))
                m_sum += M[i][j]*U[k][i]
            V[k][i] = np.dot(np.linalg.inv(lam*sigma2*I + u_sum),m_sum)
    
        m_sum = 0
        v_sum = 0
        u_sum = 0
        for item in train_data:
            ui = int(item[0])-1
            vj = int(item[1])-1
            m_sum += (item[2] - np.dot(U[k][ui],V[k][vj].reshape((-1,1))))**2
        m_sum /= 2*sigma2
    
        for u in U[k]:
            u_sum += np.linalg.norm(u)
        u_sum = u_sum*lam/2
    
        for v in V[k]:
            v_sum += np.linalg.norm(v)
        v_sum = v_sum*lam/2
    
        L[k] = L[k] - m_sum - u_sum - v_sum
    
    return L, U, V


# Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
L, U_matrices, V_matrices = PMF(train_data)

np.savetxt("objective.csv", L, delimiter=",")

np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
np.savetxt("V-50.csv", V_matrices[49], delimiter=",")
