import numpy as np
from keras.models import Model
from keras.models import load_model
import imgs,data,files

def extract_features(frame_path,model_path,out_path):
    seq_dict=imgs.read_seqs(frame_path) if(type(frame_path)==str) else frame_path
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
        name_j=name_j.split('.')[0]#+'.txt'
        out_j=out_path+'/'+name_j
        np.save(out_j,seq_j)#np.savetxt(out_j,seq_j, delimiter=',')
