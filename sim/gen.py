import numpy as np
import keras
import random
from itertools import product
import sim.dist

def get_data_generator(gen_type):
    if(type(gen_type)==tuple):
        return OneCat(gen_type[1])
    if(gen_type=="balanced"):
        return balanced_data
    return gen_data

def template(X_old,y_old,fun):
    X,y=[],[]
    n_samples=len(X_old)
    for i in range(n_samples):
        x_i,y_i=X_old[i],y_old[i]
        for x_ij,y_ij in fun(i,x_i,y_i,n_samples):
            X.append(x_ij)
            y.append(y_ij)
    X,y=np.array(X),keras.utils.to_categorical(y)
    X=[X[:,0],X[:,1]]
    return X,y  

def gen_data(X_old,y_old,n_seqs=3,n_frames=5):
    def gen_helper(i,x_i,y_i,n_samples):
        for k in range(n_seqs):
            j=random.randint(0,n_samples-1)
            x_j,y_j=X_old[j],y_old[j] 
            pairs=sim.data.sample_pairs(x_i,y_i,x_j,y_j,n_frames)
            for pair_k in pairs:
                yield pair_k
    return template(X_old,y_old,gen_helper)

#def full_data(X_old,y_old):
#    def full_helper(i,x_i,x_j,n_samples):
#        for j in range(i,n_samples):
#            yield X_old[j],y_old[j]
#    return template(X_old,y_old,full_helper)

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

def balanced_data(X_old,y_old,in_seqs=5,out_seqs=10,n_frames=5):
    dist=sim.dist.make_balanced(y_old)
    def helper(i,x_i,y_i,n_samples):
        pairs=[]
        for k in range(in_seqs):
            in_i=dist.in_cat(y_i)
            x_in,y_in=X_old[in_i],y_old[in_i] 
            pairs+=sim.dist.sample_pairs(x_i,y_i,x_in,y_in,n_frames)
        for k in range(out_seqs):
            out_i=dist.out_cat(y_i)
            x_out,y_out=X_old[out_i],y_old[out_i] 
            pairs+=sim.dist.sample_pairs(x_i,y_i,x_out,y_out,n_frames)
        for pair_k in pairs:
            yield pair_k
    return template(X_old,y_old,helper)

class OneCat(object):
    def __init__(self, cat_i):
        print("OK")
        self.cat_i = cat_i
    
    def __call__(self,X_old,y_old):
        dist=sim.dist.make_balanced(y_old)
        in_i,out_i=dist.divide(self.cat_i) 
        pairs= list(product(in_i,in_i))
        pairs+= list(product(out_i,in_i))
        X,y=[],[]
        for pair_k in pairs:
            i,j=pair_k
            x_i,y_i=X_old[i],y_old[i] 
            x_j,y_j=X_old[j],y_old[j] 
            sample_ij=sim.dist.sample_pairs(x_i,y_i,x_j,y_j,5)
            for x_t,y_t in sample_ij:
                X.append(x_t)
                y.append(y_t)
        X,y=np.array(X),keras.utils.to_categorical(y)
        X=[X[:,0],X[:,1]]
        return X,y  