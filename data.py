import numpy as np
import re
import imgs

def make_dataset(in_path):
    img_seqs=imgs.read_seqs(in_path)
    train,test=split(img_seqs.keys())
    return to_dataset(train,img_seqs),to_dataset(test,img_seqs)

def split(names,selector=None):
    if(not selector):
        selector=lambda name_i: (parse_name(name_i)[1]%2==1)
    train,test=[],[]
    for name_i in names:
        if((parse_name(name_i)[1]%2)==1):
            train.append(name_i)
        else:
            test.append(name_i)
    return train,test    

def parse_name(action_i):
    name_i=action_i.split('/')[-1]
    digits=re.findall(r'\d+',name_i)
    return int(digits[0]),int(digits[1])

def clean_name(action_i):
    name_i=action_i.split('/')[-1]
    raw=[s_i.lstrip("0") for s_i in re.findall(r'\d+',name_i)]
    return "_".join(raw)

def to_dataset(names,img_seq):
    X,y=[],[]
    for name_i in names:
        cat_i=parse_name(name_i)[0]-1
        for frame_j in img_seq[name_i]:
            X.append(frame_j)
            y.append(cat_i)
    return X,y

def get_params(X,y):
    raise Exception(X.shape)
    return count_cats(y),count_channels(X) 

def count_cats(y):
    return np.unique(np.array(y)).shape[0]

def count_channels(X):
    frame_dims=X[0].shape
    return int(frame_dims[0]/frame_dims[1])

def format_frames(frames ,n_channels=None):
    if(not n_channels):
        n_channels=count_channels(frames)
    return np.array([np.array(np.vsplit(frame_i,n_channels)).T
                      for frame_i in frames])

def seq_dataset(in_path):
    img_seqs=imgs.read_seqs(in_path)
    train,test=split(img_seqs.keys())
    X,y=[],[]
    for name_i,seq_i in img_seqs.items():
        cat_i=parse_name(name_i)[0]-1
        X.append(seq_i)
        y.append(cat_i)
    return X,y