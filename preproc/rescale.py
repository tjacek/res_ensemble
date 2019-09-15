import cv2
import imgs

def rescale(in_path,out_path,dim_x=64,dim_y=64):
    def scale_helper(frame_i):
        return cv2.resize(frame_i,(dim_x,dim_y), interpolation = cv2.INTER_CUBIC)
    imgs.transform(in_path,out_path,scale_helper)