import keras
import numpy as np
import resnet,models.ts
import random

def train(in_path,n_size=100):
    (X_train,y_train),test,params=resnet.load_data(in_path,split=True)
    X,y=gen_data(X_train,y_train)
    make_models=models.ts.get_model_factory("sim")
    model=make_models(params)
    model.fit(X,y,epochs=100,batch_size=100)

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

in_path="../MSR/sim_raw/agum"
train(in_path,n_size=10000)