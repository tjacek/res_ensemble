import numpy as np
import files

def read(in_path):
    paths=files.top_files(in_path)[:10]
    print(paths)
    return [read_binary(path_i) for path_i in paths]

def read_binary(action_path):
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