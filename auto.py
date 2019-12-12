import numpy as np
import os,keras
from keras.models import Model
from keras import backend as K
from keras.models import load_model
import data,imgs,extract,models.auto
import gc

def train(in_path,out_path=None,n_epochs=1000,recon=True):
    (X_train,y_train),(X_test,y_test)=data.make_dataset(in_path)
    n_cats,n_channels=data.get_params(X_train,y_train)
    X=data.format_frames(X_train,n_channels)
    make_model,params= models.auto.get_model_factory("basic")
    model,recon=make_model(n_channels,params)
    recon.summary()
    gc.collect()
    recon.fit(X,X,epochs=n_epochs,batch_size=256)#,
#    	shuffle=True,validation_data=(X, X))
    if(not out_path):
        dest_dir=os.path.split(in_path)[0]
        out_path=dest_dir+'/ae'
    model.save(out_path)
    if(recon):
        recon.save(out_path+"_recon")
        	
def reconstruct(in_path,model_path,out_path=None,diff=False):
    model=load_model(model_path)
    if(not out_path):
        out_path=os.path.split(in_path)[0]+'/rec'	
    def rec_helper(X):
        X=np.array(X)
        X=data.format_frames(X)
        pred= model.predict(X)
        if(diff):            	
            pred=np.abs(pred-X)
        pred=  [np.vstack(frame_i.T) for frame_i in pred]
        return pred   
    imgs.transform(in_path,out_path,rec_helper,False)

def extract_feats(in_path,model_path,out_path=None):
    if(not out_path):
        out_path=os.path.split(in_path)[0]+'/ae_feats'
    model=load_model(model_path)
    seq_dict=imgs.read_seqs(in_path) 
    feat_dict=extract.frame_features(seq_dict,model)
    extract.save_seqs(feat_dict,out_path)


#train('../smooth_time/data' )
#reconstruct("../smooth_time/data","../smooth_time/ae_recon",diff=True)
#extract_feats("../smooth_time/data","../smooth_time/ae")
#import feats
#feats.compute_feats("../smooth_time/ae_feats","../smooth_time/feats.txt")
