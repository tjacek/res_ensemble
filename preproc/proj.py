import numpy as np
import imgs,preproc.rescale

def projection(in_path,out_path):
    proj_x=get_projection(dim=0)
    proj_y=get_projection(dim=1)
    def full_projection(frames):
        x_frames=proj_x(frames)
        y_frames=proj_y(frames)
        return [np.concatenate([proj_x,proj_y],axis=0)
                    for proj_x,proj_y in zip(x_frames,y_frames)]
    imgs.transform(in_path,out_path,full_projection,False)
	
def raw_projection(in_path,out_path,dim=0):
    helper=get_helper(dim)
    imgs.transform(in_path,out_path,helper,False)

def get_projection(dim=0):
    proj_x=get_helper(dim)
    def smart_proj(frames):
        x_frames=proj_x(frames)
        x_frames=preproc.rescale.smooth_proj(x_frames)
        x_frames=preproc.rescale.scale(x_frames)
        return x_frames
    return smart_proj

def get_helper(dim=0):
    def proj_helper(frames):
        all_points=[ to_points(frame_i) for frame_i in frames]
        extr=np.array([np.amax(points_i,axis=0) for points_i in all_points])
        extr_glob=np.amax(extr,axis=0)
        max_x,max_y=extr_glob[dim],extr_glob[2]
        new_frames=[]
        for points_i in all_points:
            frame_i=np.zeros((max_x+5,max_y+5))
            for point_ij in points_i:
                x_j,y_j=int(point_ij[dim]),int(point_ij[2])
                frame_i[x_j][y_j]=200.0
            new_frames.append(frame_i)
        return new_frames
    return proj_helper

def to_points(frame_i):
    points=[]
    for cord_i in np.array(np.nonzero(frame_i)).T:
        x_i,y_i=cord_i[0],cord_i[1]
        points.append([x_i,y_i,frame_i[x_i][y_i] ])
    return np.array(points)