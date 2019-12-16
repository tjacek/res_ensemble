import keras
import numpy as np
import random
from keras.models import load_model
from keras.models import Model
import resnet,models.ts
import ens,local,files
from extract import save_seqs

def extract(frame_path,model_path,out_path=None):
    extractor=load_model(model_path)
    (X,y),names=resnet.load_data(frame_path,split=False)
    X_feats=extractor.predict(X)
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

def preproc_data(in_path,out_path,new_size=36):
    def helper(in_i,out_i):
        seqs_i=resnet.read_local_feats(in_i)
        seqs_i={name_j:local.upsampling(seq_ij,new_size) 
                 for name_j,seq_ij in seqs_i.items()}
        save_seqs(seqs_i,out_i)
    ens.template(in_path,out_path,helper)

def make_ensemble(in_path,model_path,out_path,n_epochs=10):
    files.make_dir(out_path)
    def helper(in_i,out_i):
        make_model(in_i,out_i,n_epochs)
        feat_i=out_path+"/"+out_i.split("/")[-1]
        extract(in_i,out_i,feat_i)
    ens.template(in_path,model_path,helper)
in_path="../ens/binary_seq"
#preproc_data(in_path,"../sim/imgs")
make_ensemble("../sim/imgs","../sim/models","../sim/feats")

#make_model(in_path,"sim_nn",n_epochs=50)
#extract(in_path,"../img/sim_nn","../img/feats.txt")