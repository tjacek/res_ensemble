import numpy as np
import cv2
import imgs

def rescale(in_path,out_path,dim_x=64,dim_y=64):
    imgs.transform(in_path,out_path,scale,single_frame=True)

def time(in_path,out_path):
    def pairs(seq_i):
        return [np.concatenate([seq_i[j],seq_i[j+1]]) 
                for j in range(len(seq_i)-1)] 
    imgs.transform(in_path,out_path,pairs,single_frame=False)

#def projection(in_path,out_path,dim_x=64,dim_y=64):
#    def proj_helper(frame_i):
#        proj_xz= smooth_proj(proj(frame_i,dim=True),dim_x,dim_y)
#        proj_yz= smooth_proj(proj(frame_i,dim=False),dim_x,dim_y)
#        return np.concatenate([proj_xz,proj_yz],axis=0)
#    imgs.transform(in_path,out_path,proj_helper,True)

#def proj(frame_i,dim=True):
#    points=to_points(frame_i)
#    dim=int(dim)
#    max_values=np.amax(points,axis=0)
#    max_x,max_y=max_values[dim],max_values[2]
#    proj_i=np.zeros((max_x+5,max_y+5),dtype=float)
#    for point_j in points:
#        x_j,y_j=int(point_j[dim]),int(point_j[2])
#        proj_i[x_j][y_j]=100.0
#    return proj_i

def smooth_proj(proj_i):
    if(type(proj_i)==list):
        return [smooth_proj(frame_j) for frame_j in proj_i]
    kernel2= np.ones((7,7),np.uint8)
    proj_i = cv2.dilate(proj_i,kernel2,iterations = 1)#
    proj_i[proj_i!=0]=200.0
    return proj_i

def scale(binary_img ,dim_x=64,dim_y=64):
    if(type(binary_img)==list):
        return [  scale(frame_i,dim_x,dim_y) for frame_i in binary_img]
    return cv2.resize(binary_img,(dim_x,dim_y), interpolation = cv2.INTER_CUBIC)

def remove_isol(img_i):
    kernel = np.ones((3,3),np.float32)
    kernel[1][1]=0.0
    img_i=img_i.astype(float)
    img_i[img_i!=0]=1.0
    img_i = cv2.filter2D(img_i,-1,kernel)
    img_i[ img_i<2.0]=0.0
    return img_i#binary_img