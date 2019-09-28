import numpy as np
import keras,keras.utils
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add
from keras.models import load_model
import data,files,norm,extract,feats

def extract_feats(frame_path,model_path,out_path):
    model=load_model(model_path)
    extractor=extract.make_extractor(model)
    (X,y),names=load_data(frame_path,split=False)
    X_feats=extractor.predict(X)
    feat_dict={ names[i]:feat_i for i,feat_i in enumerate(X_feats)}
    feats.save_feats(feat_dict,out_path)

def train_model(in_path,out_path=None,n_epochs=1000):
    train,test,params=load_data(in_path)
    model=make_conv(params)
    model.fit(train[0],train[1],epochs=n_epochs,batch_size=100)
    score = model.evaluate(test[0],test[1], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    if(out_path):
        model.save(out_path)

def load_data(in_path,split=True):   
    feat_dict=read_local_feats(in_path)
    if(split):
        train,test=data.split(feat_dict.keys())
        train,test=prepare_data(train,feat_dict),prepare_data(test,feat_dict)
        params={'ts_len':train[0].shape[1],'n_feats':train[0].shape[2],'n_cats': train[1].shape[1]}
        return train,test,params
    else:
        names=list(feat_dict.keys())
        return prepare_data(names,feat_dict),names

def read_local_feats(in_path):
    paths=files.top_files(in_path)
    feat_dict={}
    for path_i in paths:
        name_i=path_i.split("/")[-1]
        name_i=data.clean_name(path_i)
        feat_dict[name_i]=np.loadtxt(path_i,delimiter=",")
    return feat_dict

def prepare_data(names,feat_dict):
    X=np.array([feat_dict[name_i] for name_i in names])
    X=norm.normalize(X,'all')
    X=np.expand_dims(X,axis=-1)
    y=[data.parse_name(name_i)[0]-1 for name_i in names]
    y=keras.utils.to_categorical(y)
    return X,y

def make_conv(params):
    input_layer = Input(shape=(params['ts_len'], params['n_feats'],1))
    activ='relu' #'elu'
    pool1=add_conv_layer(input_layer,0,activ=activ)
    pool2=add_conv_layer(pool1,1,activ=activ)
    kernel_regularizer=regularizers.l1(0.001)
    hidden_layer = Dense(100,name='hidden', activation=activ,
                         kernel_regularizer=kernel_regularizer)(Flatten()(pool2))
   
    hidden_layer=BatchNormalization()(hidden_layer)
    drop1=Dropout(0.5)(hidden_layer)
    output_layer = Dense(units=params['n_cats'], activation='softmax')(drop1)
    model=Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True),
             metrics=['accuracy'])
    model.summary()
    return model

def add_conv_layer(input_layer,i=0,n_kerns=16,activ='relu',
                    kern_size=(8,1),pool_size=(4,1)):
    i=str(i)
    conv1=Conv2D(n_kerns, kernel_size=kern_size,
            activation=activ,name='conv'+i)(input_layer)
    pool1=MaxPooling2D(pool_size=pool_size,name='pool'+i)(conv1)
    return pool1#BatchNormalization()(pool1)

def concat_feats(in_path1,in_path2,out_path):
    seq_dict1,seq_dict2=read_local_feats(in_path1),read_local_feats(in_path2)
    unified_dict={ name_i:np.concatenate([ seq_i,seq_dict2[name_i]],axis=1)  
                    for name_i,seq_i in seq_dict1.items()}       
    extract.save_seqs(unified_dict,out_path)

#def train_conv(in_path,out_path=None,n_epochs=100):
#    model,(test_X,test_y)=train_model(in_path,n_epochs)
#    raw_pred=model.predict(test_X,batch_size=100)
#    pred_y,test_y=np.argmax(raw_pred,axis=1),np.argmax(test_y,axis=1)
#    print(pred_y)
#    print(test_y)
#    print(classification_report(test_y, pred_y,digits=4))
#    if(out_path):
#        model.save(out_path)