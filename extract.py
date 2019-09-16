import numpy as np
from keras.models import Model
from keras.models import load_model
import imgs,data,files

def extract_features(frame_path,model_path,out_path):
    seq_dict=imgs.read_seqs(frame_path)
    extractor=make_extractor(load_model(model_path))
    feat_dict=frame_features(seq_dict,extractor)
    save_seqs(feat_dict,out_path)

def frame_features(seq_dict,extractor):
    new_dict={}
    for name_i,seq_i in seq_dict.items():
        seq_i=data.format_frames(seq_i ,n_channels=None)
        feat_seq_i=extractor.predict(seq_i)
        new_dict[name_i]=feat_seq_i
    return new_dict

def make_extractor(model):
    return Model(inputs=model.input,
                outputs=model.get_layer("hidden").output)

def save_seqs(feat_dict,out_path):
    files.make_dir(out_path)
    for name_j,seq_j in feat_dict.items():
        name_j=name_j.split('.')[0]+'.txt'
        out_j=out_path+'/'+name_j
        np.savetxt(out_j,seq_j, delimiter=',')

#def read_seqs(in_path):
#    return { path_i.split('/')[-1]:read_dict(path_i)
#                for path_i in files.top_files(in_path)}
        
#def read_dict(dict_path):
#    return { path_i.split('/')[-1]:np.loadtxt(path_i,delimiter=',') 
#                for path_i in files.top_files(dict_path)}