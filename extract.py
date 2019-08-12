import numpy as np
from keras.models import Model
from keras.models import load_model
import data,files

def extract_features(frame_path,model_path,out_path):
    seq_dict=data.dict_dataset(frame_path)
    frame_feats=[]
    for path_i in files.top_files(model_path):
        print(path_i)
        model_i=load_model(path_i)
        extractor_i=make_extractor(model_i)
        frame_feats.append(frame_features(seq_dict,extractor_i))
    save_seqs(frame_feats,out_path)
    return frame_feats

def frame_features(data_dict,extractor):
    new_dict={}
    for name_i,seq_i in data_dict.items():
        new_dict[name_i]=extractor.predict(np.array(seq_i))
    return new_dict

def make_extractor(model):
    return Model(inputs=model.input,
                outputs=model.get_layer("hidden").output)

def save_seqs(frame_feats,out_path):
    files.make_dir(out_path)
    for i,feats_i in enumerate(frame_feats):
        out_i=out_path+'/nn'+str(i)
        files.make_dir(out_i)
        for name_j,seq_j in feats_i.items():
            out_ij=out_i+'/'+name_j
            np.savetxt(out_ij,seq_j, delimiter=',')

def read_seqs(in_path):
    return { path_i.split('/')[-1]:read_dict(path_i)
                for path_i in files.top_files(in_path)}
        
def read_dict(dict_path):
    return { path_i.split('/')[-1]:np.loadtxt(path_i,delimiter=',') 
                for path_i in files.top_files(dict_path)}

