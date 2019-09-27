import numpy as np

def normalize(X,mode='all'):
    if(mode=='seq'):
	    return [norm_seq(x_i) for x_i in X]
    return norm_all(X)

def norm_seq(x_i):
    mean_i=np.mean(x_i,axis=0)
    std_i=np.std(x_i,axis=0)
    return norm_template(x_i,mean_i,std_i)

def norm_all(X):
    new_X=np.concatenate(X)
    mean_all=np.mean(new_X,axis=0)
    std_all=np.std(new_X,axis=0)
    return [norm_template(x_i,mean_all,std_all) for x_i in X]

def norm_template(X,mean_i,std_i):
    def norm_helper(j,feature_j):
        feature_j-=mean_i[j]
        if(std_i[j]!=0):
            feature_j/=std_i[j]
        return feature_j
    return np.array([norm_helper(j,feat_j) 
                        for j,feat_j in enumerate(X.T)]).T  