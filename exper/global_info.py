import re,cv2,numpy as np
import imgs,feats

def make_global(seq_path,frame_path,out_path):
    glob_dict=clean_dict(get_global_img(seq_path))
    seq_dict=clean_dict(imgs.read_seqs(frame_path))
    def frame_fun(frame_j,name_i):
        return np.concatenate([frame_j,glob_dict[name_i]],axis=0) 	
    new_seqs={name_i:[frame_fun(frame_j,name_i) 
                for frame_j in seq_i]
                    for name_i,seq_i in seq_dict.items()}
    imgs.save_seqs(new_seqs,out_path)
      
def get_global_img(seq_path):
    seq_dict=feats.read_seqs(seq_path)
    def scale_helper(frame_j):
        return cv2.resize(frame_j,(64,64), interpolation = cv2.INTER_CUBIC)
    return {name_i:scale_helper(seq_i)
                for name_i,seq_i in seq_dict.items()} 

def clean_dict(dict_i):
	return { clean_name(name_i):data_i
		        for name_i,data_i in dict_i.items()}

def clean_name(name_i):
    numbers=re.findall(r'\d+',name_i)
    return "_".join([ n_i[1:] if n_i.startswith('0') else n_i
                        for n_i in numbers])