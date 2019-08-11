import numpy as np
import keras,keras.utils
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add
import data,norm

def train_conv(in_path,n_epochs=100):
    (train_X,train_y),(test_X,test_y)=data.img_dataset(in_path)
    params={'ts_len':train_X[0].shape[0],'n_feats':train_X[0].shape[1],
            'n_cats': max(test_y)+1}
    (train_X,train_y),(test_X,test_y)= prepare_data(train_X,train_y),prepare_data(test_X,test_y)
    print(params)
    model=make_res(params)
    model.fit(train_X,train_y,epochs=n_epochs,batch_size=32)
    raw_pred=model.predict(test_X,batch_size=32)
    pred_y,test_y=np.argmax(raw_pred,axis=1),np.argmax(test_y,axis=1)
    print(pred_y)
    print(test_y)
    print(classification_report(test_y, pred_y,digits=4))

def prepare_data(X,y):
    X=norm.normalize(X,'all')
    X= np.expand_dims(np.array(X),-1)
    y=keras.utils.to_categorical(y)
    return X,y

def make_res(params):
    input_layer = Input(shape=(params['ts_len'], params['n_feats'],1))
    activ='relu' #'elu'
    pool1=add_conv_layer(input_layer,activ=activ)
    pool2=add_conv_layer(pool1,activ=activ)

    res_layer1=add_res_layer(Flatten()(pool2),n_hidden=100,activ=activ)
    res_layer2=add_res_layer(res_layer1,n_hidden=100,activ=activ)
    
    drop1=Dropout(0.5)(res_layer2)
    output_layer = Dense(units=params['n_cats'], activation='softmax')(drop1)
    model=Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01,  momentum=0.9, nesterov=True))
    model.summary()
    return model

def make_conv(params):
    input_layer = Input(shape=(params['ts_len'], params['n_feats'],1))
    activ='relu' #'elu'
    pool1=add_conv_layer(input_layer,activ=activ)
    pool2=add_conv_layer(pool1,activ=activ)

    hidden_layer = Dense(100, activation=activ)(Flatten()(pool2))
    #kernel_regularizer=regularizers.l1(0.01)
    hidden_layer=BatchNormalization()(hidden_layer)
    drop1=Dropout(0.5)(hidden_layer)
    output_layer = Dense(units=params['n_cats'], activation='softmax')(drop1)
    model=Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01,  momentum=0.9, nesterov=True))
    model.summary()
    return model

def add_conv_layer(input_layer,n_kerns=8,activ='relu',
                    kern_size=(8,1),pool_size=(4,1)):
    conv1=Conv2D(n_kerns, kernel_size=kern_size,
            activation=activ)(input_layer)
    pool1=MaxPooling2D(pool_size=pool_size)(conv1)
    return BatchNormalization()(pool1)

def add_res_layer(input_layer,n_hidden=64,activ='relu',l1=False):
    ker_reg=regularizers.l1(0.01) if(l1) else None
    hidden_layer = Dense(n_hidden, activation=activ,kernel_regularizer=ker_reg)(input_layer)   
    hidden_layer2=Dense(n_hidden, activation=activ,kernel_regularizer=ker_reg)(hidden_layer)
    return add([hidden_layer, hidden_layer2])

train_conv("../imgset")