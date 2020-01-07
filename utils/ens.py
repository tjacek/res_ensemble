import os.path
import utils,files,feats
import sim.frames,resnet

def binary_feats(seq_path,feats_path=None):
    feats_path=get_dirs(seq_path,"feats")
    files.make_dir(feats_path)
    for i,in_i in enumerate(files.top_files(seq_path)):
        print(i)
        out_i= feats_path+'/'+in_i.split('/')[-1]
        feats.compute_feats(in_i,out_i)

def extract_ensemble(img_path,dst_dir=None):
    model_path,seq_path=get_dirs(img_path,['models','seqs'],dst_dir)
    files.make_dir(seq_path)
    for i,model_i in enumerate(files.top_files(model_path)):
        seq_i="%s/nn%d" % (seq_path,i)
        model_i="%s/nn%d" % (model_path,i)
#        resnet.extract_feats(seq_path,model_i,feat_i)
        sim.frames.extract_feats(img_path,model_i,seq_i)

def get_dirs(in_path,sufixes,dst_dir=None):
    if(not dst_dir):
        dst_dir=os.path.split(in_path)[0]
    if(type(sufixes)==str):
        return dst_dir+'/'+sufixes	
    return [dst_dir+'/'+suf_i for suf_i in sufixes]