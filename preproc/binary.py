import numpy as np
import cv2
import files

def convert(in_path,out_path):
    paths=files.top_files(in_path)
    files.make_dir(out_path)
    for path_i in paths:
        out_i= out_path +'/' +path_i.split('/')[-1]
        frames=from_binary(path_i) 
        to_imgs(frames,out_i)

def to_imgs(frames,img_path):
    files.make_dir(img_path)
    frames=standarize(frames)
    for j,frame_j in enumerate(frames):     
        frame_name_j=img_path+'/'+str(j)+".png"
        cv2.imwrite(frame_name_j,frame_j)

def standarize(frames):
    frames=np.array(frames)
    frames-= (np.amin(frames[frames!=0]) -10)
    frames/=np.amax(frames)
    frames*=240.0
    frames= np.abs(frames-250)
    frames[frames>245]=0
    return frames

def from_binary(action_path):
    print(action_path)
    with open(action_path, mode='rb') as f:
    	int_action=np.fromfile(f, dtype=np.uint32)
    header=read_header(int_action)
    assert (len(int_action)-header['size'])==3
    return read_frames(header,int_action)

def read_header(int_action):
    n_frames,width,height=int_action[0],int_action[1],int_action[2]
    frame_size,size=width*height,n_frames*width*height
    return {'n_frames':n_frames,'width':width,'height':height,'frame_size':frame_size,'size':size}

def read_frames(hd,int_action):
    indexes=range(hd['n_frames'])
    return [read_frame(i,int_action,hd) for i in indexes]

def read_frame(i,int_action,hd):
    start=3+i*hd['frame_size']
    end=start+hd['frame_size']
    frame=int_action[start:end]
    frame=np.array(frame)
    frame=frame.astype(float,copy=False)
    frame=np.reshape(frame,(hd['height'],hd['width']))
    return frame