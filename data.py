import cv2,numpy as np
import re,os
import files

def dict_dataset(in_path):
    data_dict={}
    action_dirs=files.top_files(in_path)
    for path_i in action_dirs:
        frame_paths=files.top_files(path_i)            
        seq_i=[]
        for frame_ij_path in frame_paths:
            seq_i.append(read_frame(frame_ij_path))
        name_i=path_i.split('/')
        data_dict[name_i]=seq_i  
    return data_dict
    
def frame_dataset(in_path):
    action_dirs=files.top_files(in_path)
    train,test=split(action_dirs)
    return read_frames(train)

def img_dataset(in_path):
    action_dirs=files.top_files(in_path)
    train,test=split(action_dirs)
    return read_imgs(train),read_imgs(test)

def read_frames(actions):
    X,y=[],[]
    for action_i in actions:
        cat_i,person_i= parse_name(action_i)
        frame_path=files.top_files(action_i)            
        for frame_ij_path in frame_path:
            X.append(read_frame(frame_ij_path))                
            y.append( cat_i-1)
    return np.array(X),y

def read_imgs(actions):
    X,y=[],[]
    for action_i in actions:
        cat_i,person_i= parse_name(action_i)
        X.append(cv2.imread(action_i,0).astype(float))
        y.append(cat_i-1)
    return np.array(X),y

def split(names):
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

def read_frame(frame_ij_path,n_split=4):
    frame_ij=cv2.imread(frame_ij_path,0)
    return np.array(np.vsplit(frame_ij,n_split)).T