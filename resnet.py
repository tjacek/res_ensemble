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
import data,files

def train_model(in_path):
    feat_dict=read_local_feats(in_path)
    print(len(feat_dict))

def read_local_feats(in_path):
    paths=files.top_files(in_path)
    feat_dict={}
    for path_i in paths:
        name_i=path_i.split("/")[-1]
        feat_dict[name_i]=np.loadtxt(path_i,delimiter=",")
    return feat_dict


#def extract_feats(frame_path,model_path,out_path):
#    model=load_model(model_path)
#    extractor=Model(inputs=model.input,
#                outputs=model.get_layer("hidden").output)
#    X,y,names=data.img_dataset(frame_path,split_data=False)
    #X=np.expand_dims(X,-1)
#    X,y=prepare_data(X,y)
#    X_feats=extractor.predict(X)
#    feat_dict={ names[i]:feat_i for i,feat_i in enumerate(X_feats)}
#    feats.save_feats(feat_dict,out_path)

#def train_conv(in_path,out_path=None,n_epochs=100):
#    model,(test_X,test_y)=train_model(in_path,n_epochs)
#    raw_pred=model.predict(test_X,batch_size=100)
#    pred_y,test_y=np.argmax(raw_pred,axis=1),np.argmax(test_y,axis=1)
#    print(pred_y)
#    print(test_y)
#    print(classification_report(test_y, pred_y,digits=4))
#    if(out_path):
#        model.save(out_path)

#def pretrained_model(in_path,nn_path,out_path=None,n_epochs=100):
#    (train_X,train_y),(test_X,test_y),params=load_data(in_path)
#    pretrain_model=load_model(nn_path)
#    pretrain_model.summary()
#    hidden=pretrain_model.get_layer("hidden").output
#    drop=Dropout(0.5)(hidden)
#    output=Dense(units=params['n_cats'], activation='softmax')(drop) 
#    model=Model(inputs=pretrain_model.input, outputs=output)
#    model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.SGD(lr=0.01,  momentum=0.9, nesterov=True),
#              metrics=['accuracy'])
#    model.summary()
#    model.fit(train_X,train_y,epochs=n_epochs,batch_size=100)
#    score = model.evaluate(test_X,test_y, verbose=0)
#    print('Test loss:', score[0])
#    print('Test accuracy:', score[1])
#    if(out_path):
#        model.save(out_path)

#def train_model(in_path,n_epochs):
#    (train_X,train_y),(test_X,test_y),params=load_data(in_path)
#    print(params)
#    model=make_conv(params)
#    model.fit(train_X,train_y,epochs=n_epochs,batch_size=100)
#    return model,(test_X,test_y)

#def load_data(in_path):
#    (train_X,train_y,names),(test_X,test_y,names)=data.img_dataset(in_path)
#    params={'ts_len':train_X[0].shape[0],'n_feats':train_X[0].shape[1],
#            'n_cats': max(test_y)+1}
#    return prepare_data(train_X,train_y),prepare_data(test_X,test_y),params

#def prepare_data(X,y):
#    X=norm.normalize(X,'all')
#    X= np.expand_dims(np.array(X),-1)
#    y=keras.utils.to_categorical(y)
#    return X,y

#def make_res(params):
#    input_layer = Input(shape=(params['ts_len'], params['n_feats'],1))
#    activ='relu' #'elu'
#    pool1=add_conv_layer(input_layer,activ=activ)
#    pool2=add_conv_layer(pool1,activ=activ)

    #res_layer1=add_res_layer(Flatten()(pool2),n_hidden=100,activ=activ,l1=True)
#    res_layer2=add_res_layer(Flatten()(pool2),n_hidden=64,activ=activ,l1=True,name="hidden")
    
#    drop1=Dropout(0.5)(res_layer2)
#    output_layer = Dense(units=params['n_cats'], activation='softmax')(drop1)
#    model=Model(inputs=input_layer, outputs=output_layer)
#    model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.SGD(lr=0.01,  momentum=0.9, nesterov=True))
#    model.summary()
#    return model

def make_conv(params):
    input_layer = Input(shape=(params['ts_len'], params['n_feats'],1))
    activ='relu' #'elu'
    pool1=add_conv_layer(input_layer,0,activ=activ)
    pool2=add_conv_layer(pool1,1,activ=activ)
    kernel_regularizer=regularizers.l1(0.001)
    hidden_layer = Dense(100,name='hidden', activation=activ,
                         kernel_regularizer=kernel_regularizer)(Flatten()(pool2))
   
    #hidden_layer=BatchNormalization()(hidden_layer)
    drop1=Dropout(0.5)(hidden_layer)
    output_layer = Dense(units=params['n_cats'], activation='softmax')(drop1)
    model=Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True),
             metrics=['accuracy'])
    model.summary()
    return model

#def add_conv_layer(input_layer,i=0,n_kerns=8,activ='relu',
#                    kern_size=(8,1),pool_size=(4,1)):
#    i=str(i)
#    conv1=Conv2D(n_kerns, kernel_size=kern_size,
#            activation=activ,name='conv'+i)(input_layer)
#    pool1=MaxPooling2D(pool_size=pool_size,name='pool'+i)(conv1)
#    return pool1#BatchNormalization()(pool1)

#def add_res_layer(input_layer,n_hidden=64,activ='relu',l1=False,name=None):
#    ker_reg=regularizers.l1(0.01) if(l1) else None
#    hidden_layer = Dense(n_hidden, activation=activ,kernel_regularizer=ker_reg)(input_layer)   
#    hidden_layer2=Dense(n_hidden, activation=activ,kernel_regularizer=ker_reg)(hidden_layer)
#    return add([hidden_layer, hidden_layer2],name=name)