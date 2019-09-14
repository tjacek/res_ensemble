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