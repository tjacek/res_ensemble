import cv2,numpy as np
import re,os
import files

def read_imgs(in_path,n_split=4):
    action_dirs=files.top_files(in_path)
    X,y=[],[]
    for action_i in action_dirs:
        cat_i,person_i= parse_name(action_i)
        if((person_i%2)==1):
            if(os.path.isdir(action_i)):
                frame_path=files.top_files(action_i)            
                for frame_ij_path in frame_path:
                    X.append(read_frame(frame_ij_path))                
                    y.append( cat_i)
            else:
                X.append(read_frame(action_i))
                y.append(cat_i)  
    return np.array(X),y

def parse_name(action_i):
    name_i=action_i.split('/')[-1]
    digits=re.findall(r'\d+',name_i)
    return int(digits[0]),int(digits[1])

def read_frame(frame_ij_path,n_split=4):
    frame_ij=cv2.imread(frame_ij_path,0)
    return np.array(np.vsplit(frame_ij,n_split)).T