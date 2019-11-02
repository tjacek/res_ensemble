import os.path, numpy as np
import basic,extract,feats,ens,files,imgs,persons,resnet
import preproc.agum,preproc.rescale,preproc.box

def make_frames(in_path):
    cur_dir=os.path.split(in_path)[0]
    box_path,proj_path= cur_dir+'/box',cur_dir+'/proj'
    scale_path,time_path,full_path=cur_dir+'/scale',cur_dir+'/time',cur_dir+'/full'
    if(not os.path.isdir(box_path)):
        preproc.box.box_frame(in_path,box_path)
    if(not os.path.isdir(proj_path)):
        preproc.rescale.projection(box_path,proj_path)
    if(not os.path.isdir(scale_path)):
        preproc.rescale.rescale(box_path,scale_path)
    if(not os.path.isdir(time_path)):
        preproc.rescale.time(scale_path,time_path)
    if(not os.path.isdir(full_path)):
        imgs.concat(time_path,proj_path,full_path)

def ensemble_models(frame_path,agum_path=None,n_epochs=15,model_type='exp'):
    agum_path=frame_path if(not agum_path) else agum_path
    cur_dir=os.path.split(agum_path)[0]
    nn_path,seq_path,feat_path=cur_dir+'/binary_nn',cur_dir+'/binary_seq',cur_dir+'/binary_feats'
    ens.train_binary_model(frame_path,nn_path,n_epochs=n_epochs,model_type=model_type)
    ens.binary_extract(frame_path,nn_path,seq_path)
    ens.binary_feats(seq_path,feat_path)

def person_features(frame_path,n_epochs=100):
    cur_dir=os.path.split(frame_path)[0]
    model_path=cur_dir+'/person_nn'
    feat_path=cur_dir+'/person_feats'
    persons.person_model(frame_path,model_path,n_epochs)
    persons.extract_person(frame_path,model_path,feat_path)

def ts_features(seq_path,n_epochs=1000):
    cur_dir=os.path.split(seq_path)[0]
    model_path,feat_path=cur_dir+'/ts_nn',cur_dir+'/ts_feats.txt'
    resnet.train_model(seq_path,model_path,n_epochs)
    resnet.extract_feats(seq_path,model_path,feat_path)

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

def add_mode(old_path,new_path,out_path):
    old_modes=imgs.read_seqs(old_path)
    new_modes=imgs.read_seqs(new_path)
    def add_helper(name_i):
        old_i=old_modes[name_i]
        new_i=new_modes[name_i]
        new_i=preproc.rescale.scale(new_i ,64,64)
        return np.concatenate([old_i,new_i],axis=1)
    unified={ name_i:add_helper(name_i) for name_i in list(new_modes.keys())}
    imgs.save_seqs(unified,out_path)

def ts_ensemble(seq_path,n_epochs=1000):
    cur_dir=os.path.split(seq_path)[0]
    model_path,feat_path=cur_dir+'/models',cur_dir+'/feats'
    files.make_dir(model_path)
    files.make_dir(feat_path)
    for seq_i in files.top_files(seq_path):
        id_i=seq_i.split('/')[-1]
        model_i,feat_i=model_path+'/'+id_i,feat_path+'/'+id_i
        resnet.train_model(seq_i,model_i,n_epochs)
        resnet.extract_feats(seq_i,model_i,feat_i)

def extract_ensemble(seq_path):
    cur_dir=os.path.split(seq_path)[0]
    model_path,feat_path=cur_dir+'/models',cur_dir+'/full_feats'
    files.make_dir(feat_path)
    for i,model_i in enumerate(files.top_files(model_path)):
        feat_i=feat_path+'/nn'+str(i)
        resnet.extract_feats(seq_path,model_i,feat_i)
