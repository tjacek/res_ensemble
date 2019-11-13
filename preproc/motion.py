import numpy as np,cv2
from scipy.ndimage import convolve1d
import imgs,files 

def transform(in_path,out_path,type="diff"):
    single=False
    if(type=="diff"):
        fun=diff_helper
    if(type=="motion"):
        fun=motion_helper
    if(type=="canny"):
        fun= lambda img_i:cv2.Canny(img_i,100,200)
        single=True
    if(type=="smooth"):
        fun= lambda img_i: cv2.GaussianBlur(img_i,(5,5),cv2.BORDER_DEFAULT)
        single=True
    imgs.transform(in_path,out_path,fun,single)

def motion_helper(frames):
    kernel=triangual_kernel(len(frames))
    frames=np.array(frames).T
    print(np.sum(kernel ))
    conv_frames=[convolve1d(frame_i,kernel, axis=-1,mode="constant") for frame_i in frames]
    return np.array(conv_frames).T

def action_imgs(in_path,out_path="action_imgs"):
    action_dict=imgs.read_seqs(in_path)
    action_dict=imgs.frame_tranform(preproc,action_dict)
    files.make_dir(out_path)
    action_dict={ name_i:diff(frames_i) for name_i,frames_i in action_dict.items()}
    for name_i,frames_i in action_dict.items():
        out_i=out_path+'/'+name_i.split(".")[0]+".png"
        cv2.imwrite(out_i,sum_imgs(frames_i))

def diff_helper(frames):
    return[ np.abs(frames[i]-frames[i+1])
                for i in range(len(frames)-1)]

def proj(frames):
    frames=np.array(frames)
    frames[frames!=0]=100.0
    return frames

def sum_imgs(frames):
    frames=np.array(frames)
    norm_const= 1.0/float(frames.shape[0])
    return norm_const*np.sum(frames ,axis=0)

def triangual_kernel(size):
    neg,pos=list(range(size)),list(range(size))
    pos.reverse()
    kernel=np.array(neg+[size]+pos,dtype=float )
    kernel/=np.sum(kernel)
    return 2*kernel

def parabolic_kernel(size):
    neg,pos=list(range(1,size)),list(range(1,size))
    neg.reverse()
    kernel=np.array(neg+[0]+pos,dtype=float )
    kernel/=float(size)
    kernel=(1.0 - kernel**2)
    print(kernel)
    return kernel/np.sum(kernel)

