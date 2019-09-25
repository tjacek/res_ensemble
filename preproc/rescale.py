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
        proj_xz= smooth_proj(proj(frame_i,dim=True),dim_x,dim_y)
        proj_yz= smooth_proj(proj(frame_i,dim=False),dim_x,dim_y)
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
        proj_i[x_j][y_j]=1.0
    return proj_i

def to_points(frame_i):
    points=[]
    for cord_i in np.array(np.nonzero(frame_i)).T:
        x_i,y_i=cord_i[0],cord_i[1]
        point_i=np.array([x_i,y_i,frame_i[x_i][y_i]  ])
        points.append(point_i)
    return np.array(points)

def smooth_proj(proj_i,dim_x,dim_y):
    binary_img=remove_isol(proj_i)
    kernel2= np.ones((7,7),np.uint8)
    binary_img = cv2.dilate(binary_img,kernel2,iterations = 1)
    binary_img[binary_img!=0]=200.0
    return scale(binary_img ,dim_x,dim_y)

def scale(binary_img ,dim_x,dim_y):
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