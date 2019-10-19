import keras
import numpy as np
import resnet,models.ts
import random

def train(in_path,n_size=100):
    (X_train,y_train),test,params=resnet.load_data(in_path,split=True)
    n_samples=X_train.shape[0]
    def rsample():
        return random.randint(0,n_samples-1)  
    X,y=[],[]
    for k in range(n_size): 
        i,j=rsample(),rsample()
        x_i,x_j=X_train[i],X_train[j]
        y_i,y_j=y_train[i],y_train[j]
        x_k=[x_i,x_j]#np.concatenate([x_i,x_j],axis=1)
        y_k=np.dot(y_i,y_j)
        X.append(x_k)
        y.append(y_k)
    X,y= list(zip(*X)),keras.utils.to_categorical(y)
    X=[np.array(X[0]), np.array(X[1]) ]
    make_models=models.ts.get_model_factory("sim")
    model=make_models(params)
    model.fit(X,y,epochs=100,batch_size=100)

in_path="../MSR/sim_raw/agum"
train(in_path,n_size=10000)