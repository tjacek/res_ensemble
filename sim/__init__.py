#import keras
import numpy as np,cv2
#from keras.models import load_model
#from keras.models import Model
import resnet,models.ts
import ens,local,files
from extract import save_seqs
import data,sim.gen

def show_frames(in_path,out_path):
    X_train,y_train=data.seq_dataset(in_path)
    X,y=sim.gen.gen_data(X_train,y_train)
    files.make_dir(out_path)
    for i,y_i in enumerate(y):
        x0,x1=X[i]
        cat_i=np.argmax(y_i)
        img_i=np.concatenate([x0,x1])
        out_i='%s/%d_%d.png' % (out_path,i,cat_i)
        print(out_i)
        cv2.imwrite(out_i,img_i)

def extract(frame_path,model_path,out_path=None):
    extractor=load_model(model_path)
    (X,y),names=resnet.load_data(frame_path,split=False)
    X_feats=extractor.predict(X)
    resnet.get_feat_dict(X_feats,names,out_path)

def make_model(in_path,out_path=None,n_epochs=50):
    (X_train,y_train),test,params=resnet.load_data(in_path,split=True)
    X,y=full_data(X_train,y_train)
    make_models=models.ts.get_model_factory("sim_exp")
    sim_metric,model=make_models(params)
    sim_metric.fit(X,y,epochs=n_epochs,batch_size=100)
    if(out_path):
        model.save(out_path)

def preproc_data(in_path,out_path,new_size=36,single=False):
    def helper(in_i,out_i):
        seqs_i=resnet.read_local_feats(in_i)
        seqs_i={name_j:local.upsampling(seq_ij,new_size) 
                 for name_j,seq_ij in seqs_i.items()}
        save_seqs(seqs_i,out_i)
    if(single):
        helper(in_path,out_path)
    else:
        ens.template(in_path,out_path,helper)

def make_ensemble(in_path,model_path,out_path,n_epochs=50):
    files.make_dir(out_path)
    def helper(in_i,out_i):
        make_model(in_i,out_i,n_epochs)
        feat_i=out_path+"/"+out_i.split("/")[-1]
        extract(in_i,out_i,feat_i)
    ens.template(in_path,model_path,helper)

#in_path="../time/sim2/seq"
#preproc_data(in_path,"../time/imgs",128,single=True)
#make_ensemble("../L2_sim/imgs","../L2_sim/models","../L2_sim/feats")

#make_model(in_path,"../time/sim2/sim_nn",n_epochs=50)
#extract(in_path,"../time/sim2/sim_nn","../time/sim2/feat")