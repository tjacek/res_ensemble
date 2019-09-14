import cv2
import files

def read_seqs(in_path):
    seqs={}
    for seq_path_i in files.top_files(in_path):
        frames=[ cv2.imread(frame_path_j, cv2.cv2.IMREAD_GRAYSCALE)
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