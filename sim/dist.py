import numpy as np
import random
from collections import defaultdict

class BalancedDist(object):
    def __init__(self,by_cat):
        self.by_cat=by_cat
        self.n_cats=len(self.by_cat)

    def in_cat(self,cat_i):
        return np.random.choice(self.by_cat[cat_i])

    def out_cat(self,cat_i):
        j=random.randint(0,self.n_cats-2)
        if(j>=cat_i):
       	    j+=1
        return self.in_cat(j)

def make_balanced(y):
    by_cat=defaultdict(lambda :[])
    for i,y_i in enumerate(y):
        by_cat[y_i].append(i)
    return BalancedDist(by_cat)

def sample_pairs(x_i,y_i,x_j,y_j,n_frames):
    indexes_i=sample_seq(x_i,n_frames)
    indexes_j=sample_seq(x_j,n_frames)
    y_k=int(y_i==y_j)
    return [ ((x_i[t_i],x_j[t_j]),y_k) 
                for t_i,t_j in zip(indexes_i,indexes_j)]

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
    dist=dist**2
    dist/=np.sum(dist)
    return dist