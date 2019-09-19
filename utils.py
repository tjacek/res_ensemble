import os.path
import basic,extract,feats

def single_model(frame_path):
    cur_dir=os.path.split(frame_path)[0]
    nn_path,seq_path,feat_path=cur_dir+'/nn',cur_dir+'/seq',cur_dir+'/feats'
    basic.simple_exp(frame_path,nn_path,n_epochs=150)
    extract.extract_features(frame_path,nn_path,seq_path)
    feats.compute_feats(seq_path,feat_path)	