import os.path
import utils,files,feats,local
import sim.frames,resnet,extract#,imgs

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

def upsample(seq_path,new_size=128):
    spline_path=get_dirs(seq_path,'spline')
    files.make_dir(spline_path)
    for i,in_i in enumerate(files.top_files(seq_path)): 
        seq_dict=resnet.read_local_feats(in_i)
        seq_dict={name_i:local.upsampling(seq_i,new_size) 
                    for name_i,seq_i in seq_dict.items()}
        out_i="%s/nn%d" % (spline_path,i)
        extract.save_seqs(seq_dict,out_i)

def ts_ensemble(seq_path,n_epochs=1000):
    model_path,feat_path=get_dirs(seq_path,['ts_models','ts_feats'])
    files.make_dir(model_path)
    files.make_dir(feat_path)
    for seq_i in files.top_files(seq_path):
        id_i=seq_i.split('/')[-1]
        model_i,feat_i=model_path+'/'+id_i,feat_path+'/'+id_i
        resnet.train_model(seq_i,model_i,n_epochs)
        resnet.extract_feats(seq_i,model_i,feat_i)

#def ts_ensemble(seq_path,n_epochs=1000,single=False):
#    cur_dir=os.path.split(seq_path)[0]
#    if(single):
#        model_path,feat_path=cur_dir+'/nn',cur_dir+'/feat'
#        resnet.train_model(seq_path,model_path,n_epochs)
#        resnet.extract_feats(seq_path,model_path,feat_path)
#        return