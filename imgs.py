import cv2
import files

def transform(in_path,out_path,frame_fun,single_frame=False):
    img_seqs=read_seqs(in_path)
    if(single_frame):
        new_seqs={ name_i:[frame_fun(frame_j) for frame_j in seq_i]
                    for name_i,seq_i in img_seqs.items()}
    else:
        new_seqs={ name_i: frame_fun(seq_i)
                    for name_i,seq_i in img_seqs.items()}
    save_seqs(new_seqs,out_path)

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
