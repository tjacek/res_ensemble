import numpy as np
import keras
import random

def gen_data(X_old,y_old,n_seqs=3,n_frames=5):
    def gen_helper(i,x_i,y_i,n_samples):
        for k in range(n_seqs):
            j=random.randint(0,n_samples-1)
            x_j,y_j=X_old[j],y_old[j] 
            indexes_i=sample_seq(x_i,n_frames)
            indexes_j=sample_seq(x_j,n_frames)
            y_k=int(y_i==y_j)
            for t_i,t_j in zip(indexes_i,indexes_j):
                frame_i,frame_j=x_i[t_i],x_j[t_j]
                yield (frame_i,frame_j),y_k
    return template(X_old,y_old,gen_helper)

def full_data(X_old,y_old):
    def full_helper(i,x_i,x_j,n_samples):
        for j in range(i,n_samples):
            yield X_old[j],y_old[j]
    return template(X_old,y_old,full_helper)

def rand_data(X_old,y_old,size=10):
    def rand_helper(i,x_i,y_i,n_samples):
        print(i)
        j=random.randint(0,n_samples-1) 
        x_j,y_j=X_old[j],y_old[j] 
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

def sample_seq(frames,size=5):
    n_frames=len(frames)
    dist=get_dist(n_frames)
    def sample(n):   
        return np.random.choice(np.arange(n_frames),n,p=dist)
    return sample(size)

def get_dist(n):
    inc,dec=np.arange(n),np.flip(np.arange(n))
    dist=np.amin(np.array([inc,dec]),axis=0)
    dist=dist.astype(float)
    dist/=np.sum(dist)
    return dist