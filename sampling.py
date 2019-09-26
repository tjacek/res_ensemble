import numpy as np
import imgs

def sample_imgs(in_path,out_path):
    imgs.transform(in_path,out_path,sample_seq)

def sample_seq(frames):
    n_frames=len(frames)
    dist=get_dist(n_frames)    
    indexes=np.random.choice(np.arange(n_frames),n_frames,p=dist)
    return [frames[i] for i in indexes]

def get_dist(n):
    inc,dec=np.arange(n),np.flip(np.arange(n))
    diff=np.abs(inc - dec)
    dist=1.0/(1.0+diff)
    return dist/np.sum(dist)