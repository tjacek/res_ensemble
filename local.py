import numpy as np
import imgs,files

class Extractor(object):
    def __init__(self):
        self.fun=[area,max_z]

    def __call__(self,frame_i):
        frame_feats=[]
        for fun_j in self.fun:
            frame_feats+=fun_j(frame_i)
        return frame_feats

def compute(in_path,out_path):
    seq_dict=imgs.read_seqs(in_path)
    extract=Extractor()
    files.make_dir(out_path)
    for name_i,seq_i in seq_dict.items():
        feat_seq_i=np.array([extract(frame_i) for frame_i in seq_i])
        name_i=name_i.split('.')[0]+'.txt'
        out_i=out_path+'/'+name_i
        np.savetxt(out_i,feat_seq_i,delimiter=',')

def area(frame_i):
    frame_i[frame_i!=0]=1.0
    return [np.mean(frame_i)]

def max_z(frame_i):
    max_cord=np.unravel_index(frame_i.argmax(), frame_i.shape)
    return [(max_cord[0]/frame_i.shape[0]), (max_cord[1]/frame_i.shape[1])]
