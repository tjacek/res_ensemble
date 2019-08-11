import numpy as np
import keras,keras.utils
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report
from keras import regularizers
import data,norm

def train_conv(in_path,n_epochs=100):
    (train_X,train_y),(test_X,test_y)=data.img_dataset(in_path)
    params={'ts_len':train_X[0].shape[0],'n_feats':train_X[0].shape[1],
            'n_cats': max(test_y)+1}
    (train_X,train_y),(test_X,test_y)= prepare_data(train_X,train_y),prepare_data(test_X,test_y)
    print(params)
    model=make_conv(params)
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

def make_conv_(params):
    input_layer = Input(shape=(params['ts_len'], params['n_feats'],1))
    conv1=Conv2D(8, kernel_size=(8, 1),
            activation='relu')(input_layer)
    pool1=MaxPooling2D(pool_size=(4, 1))(conv1)
    conv2=Conv2D(8, kernel_size=(8, 1),
              activation='relu',
              input_shape=(params['ts_len'],params['n_feats'],1))(pool1)
    pool2=MaxPooling2D(pool_size=(4, 1))(conv2)
    hidden_layer = Dense(64, activation='relu')(Flatten()(pool2))
    drop1=Dropout(0.5)(hidden_layer)
    output_layer = Dense(units=params['n_cats'], activation='softmax')(drop1)
    model=Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01,  momentum=0.9, nesterov=True))
    return model

train_conv("../imgset")