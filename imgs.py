import cv2,numpy as np
import files

class Pipeline(object):
    def __init__(self,transforms):
        self.transforms=transforms

    def __call__(self,frame_i):
        frame_i=self.transforms[0](frame_i)
        for transform_j in self.transforms[1:]:
            frame_i=transform_j(frame_i)
        return frame_i

def transform(in_path,out_path,frame_fun,single_frame=False):
    if(type(frame_fun)==list):
        frame_fun=Pipeline(frame_fun)
    img_seqs=read_seqs(in_path)
    if(single_frame):
        new_seqs=frame_tranform(frame_fun,img_seqs)
    else:
        new_seqs={ name_i: frame_fun(seq_i)
                    for name_i,seq_i in img_seqs.items()}
    save_seqs(new_seqs,out_path)

def frame_tranform(frame_fun,img_seqs):
    return { name_i:[frame_fun(frame_j) for frame_j in seq_i]
                    for name_i,seq_i in img_seqs.items()}

def read_seqs(in_path):
    seqs={}
    for seq_path_i in files.top_files(in_path):
        frames=[ cv2.imread(frame_path_j, cv2.IMREAD_GRAYSCALE)
                    for frame_path_j in files.top_files(seq_path_i)]
        name_i=seq_path_i.split('/')[-1]
        print(name_i)
        seqs[name_i]=frames
    return seqs	

def save_seqs(seq_dict,out_path):
    files.make_dir(out_path)
    for name_i,seq_i in seq_dict.items():
        seq_path_i=out_path+'/'+name_i
        files.make_dir(seq_path_i)
        for j,frame_j in enumerate(seq_i):     
            frame_name_j=seq_path_i+'/'+str(j)+".png"
            cv2.imwrite(frame_name_j,frame_j)

def concat(in_path1,in_path2,out_path):
    seq1,seq2=read_seqs(in_path1),read_seqs(in_path2)
    names=seq1.keys()
    concat_seqs={}
    for name_i in names:
        seq1_i,seq2_i=seq1[name_i],seq2[name_i]
        seq_len=min(len(seq1_i),len(seq2_i))
        seq1_i,seq2_i= seq1_i[:seq_len],seq2_i[:seq_len]
        new_seq_i=np.concatenate( [seq1_i,seq2_i],axis=1)
        concat_seqs[name_i]=new_seq_i
    save_seqs(concat_seqs,out_path)