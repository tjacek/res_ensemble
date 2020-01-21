import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import skew,pearsonr
import imgs,files

class Extractor(object):
    def __init__(self):
        self.fun=[area,max_z]
        self.points=[moments,corl]

    def __call__(self,frame_i):
        frame_feats=[]
        if(self.fun):
            for fun_j in self.fun:
                frame_feats+=fun_j(frame_i)
        points=nonzero_points(frame_i)
        if(self.points):
            for fun_j in self.points:
                frame_feats+=fun_j(points)
        return frame_feats

def nonzero_points(frame_i):
    xy_nonzero=np.nonzero(frame_i)
    z_nozero=frame_i[xy_nonzero]
    xy_nonzero,z_nozero=np.array(xy_nonzero),np.expand_dims(z_nozero,axis=0)
    return np.concatenate([xy_nonzero,z_nozero],axis=0)

def compute(in_path,out_path,upsample=False):
    seq_dict=imgs.read_seqs(in_path)
    extract=Extractor()
    files.make_dir(out_path)
    for name_i,seq_i in seq_dict.items():
        feat_seq_i=np.array([extract(frame_i) for frame_i in seq_i])
        name_i=name_i.split('.')[0]+'.txt'
        out_i=out_path+'/'+name_i
        if(upsample):
            feat_seq_i=upsampling(feat_seq_i)
        np.savetxt(out_i,feat_seq_i,delimiter=',')

def area(frame_i):
    return [np.count_nonzero(frame_i)/np.prod(frame_i.shape)]

def max_z(frame_i):
    max_cord=np.unravel_index(frame_i.argmax(), frame_i.shape)
    return [(max_cord[0]/frame_i.shape[0]), (max_cord[1]/frame_i.shape[1])]

def moments(points):
    std_i=list(np.std(points,axis=1))
    skew_i=list(skew(points,axis=1))
    return std_i+skew_i

def corl(points):
    x,y,z=points[0],points[1],points[2]
    return [pearsonr(x,y)[0],pearsonr(z,y)[0],pearsonr(x,z)[0]]

def upsampling(seq_j,new_size=128):
    feats=[spline(feat_i,new_size) for feat_i in np.array(seq_j).T]
    return np.array(feats).T

def spline(feat_i,new_size=128):
    old_size=feat_i.shape[0]
    old_x=np.arange(old_size).astype(float)  
    step=float(new_size)/float(old_size)
    old_x*=step     
    cs=CubicSpline(old_x,feat_i)
    new_size=np.arange(new_size)  
    return cs(new_size)

compute("../MSR/box","../MSR/deep/full")