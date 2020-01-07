import os.path
import utils,files,resnet
import sim.frames

def extract_ensemble(img_path,dst_dir=None):
    if(not dst_dir):
        dst_dir=os.path.split(img_path)[0]
    model_path,feat_path=dst_dir+'/models',dst_dir+'/seqs'
    files.make_dir(feat_path)
    for i,model_i in enumerate(files.top_files(model_path)):
        seq_i="%s/nn%d" % (feat_path,i)
        model_i="%s/nn%d" % (model_path,i)
#        resnet.extract_feats(seq_path,model_i,feat_i)
        sim.frames.extract_feats(img_path,model_i,seq_i)