import os.path
import basic,extract,feats,ens,files
import preproc.agum,preproc.rescale

def ensemble_models(frame_path,agum_path=None,n_epochs=15):
    agum_path=frame_path if(not agum_path) else agum_path
    cur_dir=os.path.split(agum_path)[0]
    nn_path,seq_path,feat_path=cur_dir+'/binary_nn',cur_dir+'/binary_seq',cur_dir+'/binary_feats'
    ens.train_binary_model(frame_path,nn_path,n_epochs=n_epochs)
    ens.binary_extract(frame_path,nn_path,seq_path)
    ens.binary_feats(seq_path,feat_path)

def extract(frame_path,nn_path,seq_path,k=15):
    nn_paths=files.top_files(nn_path)[k:]
    print(nn_paths)
    ens.binary_extract(frame_path,nn_paths,seq_path)

def single_model(frame_path,n_epochs,nn_type):
    cur_dir=os.path.split(frame_path)[0]
    nn_path,seq_path,feat_path=cur_dir+'/nn',cur_dir+'/seq',cur_dir+'/feats'
    basic.simple_exp(frame_path,nn_path,n_epochs=n_epochs,model_type=nn_type)
    extract.extract_features(frame_path,nn_path,seq_path)
    feats.compute_feats(seq_path,feat_path)	

def agum_seq(in_path):
    agum_path=in_path+"_agum"
    scale_path=in_path+"_agum_scale"
    time_path=in_path+"_agum_time"
    preproc.agum.agum_data(in_path,agum_path)
    preproc.rescale.rescale(agum_path,scale_path)
    preproc.rescale.pairs(scale_path,time_path)