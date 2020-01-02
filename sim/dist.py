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

y=[1,1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]
dist=make_balanced(y)
print(dist.out_cat(0))