import numpy as np
import keras.utils
from keras.models import load_model
import imgs,data,basic,models,extract,local

def extract_person(frame_path,model_path,out_path):
    seq_dict=imgs.read_seqs(frame_path)
    model=load_model(model_path)
    feat_dict={} 
    for name_i,seq_i in seq_dict.items():
        seq_i= data.format_frames(seq_i)
        feat_dict[name_i]=local.upsampling( model.predict(seq_i))
    extract.save_seqs(feat_dict,out_path)

def person_model(in_path,out_path,n_epochs=100):
    seq_dict=imgs.read_seqs(in_path)
    train,test=data.split(seq_dict.keys())
    persons=[data.parse_name(name_i)[1]-1 for name_i in train]
    persons=keras.utils.to_categorical(persons)
    X,y=to_dataset(train,seq_dict)
    n_cats,n_channels=y.shape[1],X.shape[-1]
    model=models.make_exp(n_cats,n_channels)
    model.summary()
    model.fit(X,y,epochs=n_epochs,batch_size=256)
    model.save(out_path)

def to_dataset(names,img_seq):
    X,y=[],[]
    for name_i in names:
        cat_i=data.parse_name(name_i)[1]-1
        cat_i=int(cat_i/2)
        for frame_j in img_seq[name_i]:
            X.append(frame_j)
            y.append(cat_i)
    X=data.format_frames(X)
    return np.array(X),keras.utils.to_categorical(y)