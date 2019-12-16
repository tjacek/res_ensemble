import keras
import numpy as np
import random
import gc
from keras.models import load_model
from keras.models import Model
import resnet,models.ts
import ens,local

def extract(frame_path,model_path,out_path=None):
    extractor=load_model(model_path)
    (X,y),names=resnet.load_data(frame_path,split=False)
    X_feats=extractor.predict(X)#[X,X])
    resnet.get_feat_dict(X_feats,names,out_path)

def make_model(in_path,out_path=None,n_epochs=50):
    (X_train,y_train),test,params=resnet.load_data(in_path,split=True)
    X,y=full_data(X_train,y_train)
    make_models=models.ts.get_model_factory("sim")
    sim_metric,model=make_models(params)
    sim_metric.fit(X,y,epochs=n_epochs,batch_size=100)
    if(out_path):
        model.save(out_path)

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

def preproc_data(in_path,out_path):
    def helper(in_i,out_i):
        seq_i=resnet.read_local_feats(in_i)
        print(len(seq_i))
        print(out_i)
    ens.template(in_path,out_path,helper)

in_path="../ens/binary_seq"
preproc_data(in_path,"../sim/imgs")
#make_model(in_path,"sim_nn",n_epochs=50)
#extract(in_path,"../img/sim_nn","../img/feats.txt")
