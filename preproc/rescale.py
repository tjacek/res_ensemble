import numpy as np
import cv2
import imgs

def rescale(in_path,out_path,dim_x=64,dim_y=64):
    def scale_helper(seq_i):
        return [cv2.resize(frame_j,(dim_x,dim_y), interpolation = cv2.INTER_CUBIC)
                    for frame_j in seq_i] 
    imgs.transform(in_path,out_path,scale_helper)

def pairs(in_path,out_path):
    def pair_helper(seq_i):
        return [np.concatenate([seq_i[j],seq_i[j+1]]) 
                    for j in range(len(seq_i)-1)] 
    imgs.transform(in_path,out_path,pair_helper)