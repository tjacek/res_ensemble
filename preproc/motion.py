import numpy as np,cv2
import imgs,files

def action_imgs(in_path,out_path="action_imgs"):
    action_dict=imgs.read_seqs(in_path)
    files.make_dir(out_path)
    action_dict={ name_i:diff(frames_i) for name_i,frames_i in action_dict.items()}
    for name_i,frames_i in action_dict.items():
        out_i=out_path+'/'+name_i.split(".")[0]+".png"
        cv2.imwrite(out_i,sum_imgs(frames_i))

def diff(frames):
    return[ np.abs(frames[i]-frames[i+1])
                for i in range(len(frames)-1)]

def sum_imgs(frames):
    frames=np.array(frames)
    norm_const= 1.0/float(frames.shape[0])
    return norm_const*np.sum(frames ,axis=0)