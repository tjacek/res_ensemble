import numpy as np
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
    n_samples=X_old.shape[0]
    X,y=[],[]
    for i in range(n_samples):
        x_i,y_i=X_old[i],y_old[i]
        for j in range(i,n_samples):
            x_j,y_j=X_old[j],y_old[j]
            X.append([x_i,x_j])
            y.append(np.dot(y_i,y_j))
    X,y=np.array(X),keras.utils.to_categorical(y)
    X=[X[:,0],X[:,1]]
    return X,y

def random_data(X_old,y_old,size=100):
    n_samples=X_old.shape[0]
    X,y=[],[]
    for i in range(n_samples):
        x_i,y_i=X_old[i],y_old[i]
        indexes=[random.randint(0,n_samples-1) 
                    for j in range(size)]
        for j in indexes:
            x_j,y_j=X_old[j],y_old[j]
            X.append([x_i,x_j])
            y.append(np.dot(y_i,y_j))
    X,y=np.array(X),keras.utils.to_categorical(y)
    X=[X[:,0],X[:,1]]
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