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

def projection(in_path,out_path,dim_x=64,dim_y=64):
    def proj_helper(frame_i):
        proj_xz= cv2.resize(proj(frame_i,dim=True),(dim_x,dim_y), interpolation = cv2.INTER_CUBIC)
        proj_yz= cv2.resize(proj(frame_i,dim=False),(dim_x,dim_y), interpolation = cv2.INTER_CUBIC)
        return np.concatenate([proj_xz,proj_yz],axis=0)
    imgs.transform(in_path,out_path,proj_helper,True)

def proj(frame_i,dim=True):
    points=to_points(frame_i)
    dim=int(dim)
    max_values=np.amax(points,axis=0)
    max_x,max_y=max_values[dim],max_values[2]
    proj_i=np.zeros((max_x+5,max_y+5))
    for point_j in points:
        x_j,y_j=int(point_j[dim]),int(point_j[2])
        proj_i[x_j][y_j]=200.0
    return proj_i

def to_points(frame_i):
    points=[]
    for cord_i in np.array(np.nonzero(frame_i)).T:
        x_i,y_i=cord_i[0],cord_i[1]
        point_i=np.array([x_i,y_i,frame_i[x_i][y_i]  ])
        points.append(point_i)
    return np.array(points)