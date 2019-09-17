import numpy as np
import scipy.stats
import extract,files

def compute_feats(in_path,out_path):
    data_dict=read_seqs(in_path)
    feat_dict={ name_i:feat_vector(seq_i)
                    for name_i,seq_i in data_dict.items()}
    save_feats(feat_dict,out_path)

def feat_vector(seq_j):
    feats=[]
    for ts_k in seq_j.T:
    	feats+=EBTF(ts_k)
    return np.array(feats)

def EBTF(feat_i):
    if(np.all(feat_i==0)):
        return [0.0,0.0,0.0,0.0]
    return [np.mean(feat_i),np.std(feat_i),
    	                scipy.stats.skew(feat_i),time_corl(feat_i)]

def time_corl(feat_i):
    n_size=feat_i.shape[0]
    x_i=np.arange(float(n_size),step=1.0)#1.0,step=step)
    return scipy.stats.pearsonr(x_i,feat_i)[0]

def save_feats(feat_dict,out_path):
    lines=[]
    for name_j,feat_j in feat_dict.items():
        line_i=np.array2string(feat_j,separator=",")#,precision=decimals)
        line_i=line_i.replace('\n',"")+'#'+name_j
        lines.append(line_i)
    feat_txt='\n'.join(lines)
    feat_txt=feat_txt.replace('[','').replace(']','')
    file_str = open(out_path,'w')
    file_str.write(feat_txt)
    file_str.close()


def read_seqs(in_path):
    return { path_i.split('/')[-1]:np.loadtxt(path_i,delimiter=',') 
                for path_i in files.top_files(in_path)}