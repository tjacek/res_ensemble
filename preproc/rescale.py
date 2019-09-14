import cv2
import imgs

def rescale(in_path,out_path,dim_x=64,dim_y=64):
    img_seqs=imgs.read_seqs(in_path)
    def scale_helper(frame_i):
        return cv2.resize(frame_i,(dim_x,dim_y), interpolation = cv2.INTER_CUBIC)
    rescaled_seqs={ name_i:[scale_helper(frame_j) for frame_j in seq_i] 
                        for name_i,seq_i in img_seqs.items()}
    imgs.save_seqs(rescaled_seqs,out_path)