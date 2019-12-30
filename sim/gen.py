import numpy as np
import keras
import random

def gen_data(X_old,y_old):
    n_samples,n_cats=X_old.shape[0],y_old.shape[1]
    X,y=[],[]
    for i in range(n_samples):
        x_i,y_i=X_old[i],y_old[i]
        for j in range(n_cats):
            rn=random.randint(0,n_samples-1) 
            x_j,y_j=X_old[rn],y_old[rn]
            X.append([x_i,x_j])
            y.append(np.dot(y_i,y_j))
    X,y=np.array(X),keras.utils.to_categorical(y)
    X=[X[:,0],X[:,1]]
    return X,y

def full_data(X_old,y_old):
    def full_helper(i,x_i,x_j,n_samples):
        for j in range(i,n_samples):
            yield X_old[j],y_old[j]
    return template(X_old,y_old,full_helper)

def rand_data(X_old,y_old,size=10):
#    raise Exception( type(X_old[0]))
    def rand_helper(i,x_i,y_i,n_samples):
        print(i)
        j=random.randint(0,n_samples-1) 
        x_j,y_j=X_old[j],y_old[j] 
#        raise Exception(len(x_j))
        for k in range(size):
            ik=random.randint(0,len(x_i)-1)
            jk=random.randint(0,len(x_j)-1) 
            yk=int(y_i==y_j)
            yield (x_j[jk],x_i[ik]),yk
    return template(X_old,y_old,rand_helper)

def template(X_old,y_old,fun):
    X,y=[],[]
    n_samples=len(X_old)
    for i in range(n_samples):
        x_i,y_i=X_old[i],y_old[i]
        for x_ij,y_ij in fun(i,x_i,y_i,n_samples):
            X.append(x_ij)
            y.append(y_ij)
    X,y=np.array(X),keras.utils.to_categorical(y)
#    X=[X[:,0],X[:,1]]
    return X,y    

def sample_seq(frames):
    n_frames=len(frames)
    dist=get_dist(n_frames)
    def sample(n):   
        return np.random.choice(np.arange(n_frames),n,p=dist)
    return sample

def get_dist(n):
    inc,dec=np.arange(n),np.flip(np.arange(n))
    return np.amin(np.array([inc,dec]),axis=0)