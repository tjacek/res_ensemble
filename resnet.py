import numpy as np
import keras,keras.utils
import data,files,norm,extract,feats
import models.ts
from keras.models import load_model

def extract_feats(frame_path,model_path,out_path=None):
    model=load_model(model_path)
    extractor=extract.make_extractor(model)
    (X,y),names=load_data(frame_path,split=False)
    X_feats=extractor.predict(X)
    return get_feat_dict(X_feats,names,out_path)

def get_feat_dict(X_feats,names,out_path):
    feat_dict={ names[i]:feat_i for i,feat_i in enumerate(X_feats)}
    if(out_path):
        feats.save_feats(feat_dict,out_path)
    return feat_dict

def train_model(in_path,out_path=None,n_epochs=1000,model_type="old"):
    train,test,params=load_data(in_path)
    make_models=models.ts.get_model_factory(model_type)
    model=make_models(params)
    model.fit(train[0],train[1],epochs=n_epochs,batch_size=100)
    score = model.evaluate(test[0],test[1], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    if(out_path):
        model.save(out_path)

def load_data(in_path,split=True):   
    feat_dict=read_local_feats(in_path)
    if(split):
        train,test=data.split(feat_dict.keys())
        train,test=prepare_data(train,feat_dict),prepare_data(test,feat_dict)
        params={'ts_len':train[0].shape[1],'n_feats':train[0].shape[2],'n_cats': train[1].shape[1]}
        return train,test,params
    else:
        names=list(feat_dict.keys())
        return prepare_data(names,feat_dict),names

def read_local_feats(in_path):
    paths=files.top_files(in_path)
    feat_dict={}
    for path_i in paths:
        name_i=path_i.split("/")[-1]
        name_i=data.clean_name(path_i)
        feat_dict[name_i]=np.loadtxt(path_i,delimiter=",")
    return feat_dict

def prepare_data(names,feat_dict):
    X=np.array([feat_dict[name_i] for name_i in names])
    X=norm.normalize(X,'all')
    X=np.expand_dims(X,axis=-1)
    y=[data.parse_name(name_i)[0]-1 for name_i in names]
    y=keras.utils.to_categorical(y)
    return X,y

def concat_feats(in_path1,in_path2,out_path):
    seq_dict1,seq_dict2=read_local_feats(in_path1),read_local_feats(in_path2)
    unified_dict={}
    for name_i,seq_i in seq_dict1.items():
        seq1,seq2=equal_seqs(seq_i,seq_dict2[name_i])
        unified_dict[name_i]=np.concatenate([ seq1,seq2],axis=1)                      
    extract.save_seqs(unified_dict,out_path)

def equal_seqs(seq1,seq2):
    seq_len=min(seq1.shape[0],seq2.shape[0])
    return seq1[:seq_len],seq2[:seq_len]