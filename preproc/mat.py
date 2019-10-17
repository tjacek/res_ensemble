import numpy as np
import scipy.io
import cv2
import files
from preproc.binary import standarize 

def convert_inert(in_path,out_path):
    paths=files.top_files(in_path)
    files.make_dir(out_path)
    for path_i in paths:
        out_i= out_path +'/' +path_i.split('/')[-1]
        mat_i = scipy.io.loadmat(path_i)
        mat_i=mat_i['d_iner']
        np.savetxt(out_i,mat_i,delimiter=',')

def convert(in_path,out_path):
    paths=files.top_files(in_path)
    files.make_dir(out_path)
    for path_i in paths:
        print(path_i)
        out_i= out_path +'/' +path_i.split('/')[-1]
        files.make_dir(out_i)
        mat_i = scipy.io.loadmat(path_i)
        seq_i=mat_i['d_depth']
        for j,frame_j in enumerate(seq_i.T):
            frame_name_j=out_i+'/'+str(j)+".png"
            frame_j= standarize(frame_j.T) 
            cv2.imwrite(frame_name_j,frame_j)