#import cv2,numpy as np
import re,os
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

def to_dataset(names,img_seq):
    X,y=[],[]
    for name_i in names:
        cat_i=parse_name(name_i)[0]-1
        for frame_j in img_seq[name_i]:
            X.append(frame_j)
            y.append(cat_i)
    return X,y