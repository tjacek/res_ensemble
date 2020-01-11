import resnet
from keras.models import Model,Sequential
from keras.layers import Input,LSTM,Dense,Activation,Flatten,Dropout,Bidirectional,GlobalAveragePooling1D
from keras.layers.convolutional import Conv3D,MaxPooling1D,Conv1D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import numpy as np
from keras.models import load_model

import keras
from keras.layers import Conv2D,MaxPooling2D

def extract_features(frame_path,model_path,out_path):
    (X,y),names=resnet.load_data(frame_path,split=False)
    X=np.squeeze(X)
    model=load_model(model_path)
    extractor=Model(inputs=model.input,
                outputs=model.get_layer("hidden").output)
    X_feats=extractor.predict(X)
    return resnet.get_feat_dict(X_feats,names,out_path)

def make_model(in_path,out_path=None,n_epochs=50):
    (X_train,y_train),(X_test,y_test),params=resnet.load_data(in_path,split=True)
    X_train,X_test=np.squeeze(X_train),np.squeeze(X_test)
    model=lstm_model(params)
    model.fit(X_train,y_train,epochs=n_epochs,batch_size=100)
    if(out_path):
        model.save(out_path)
    results = model.evaluate(X_test, y_test, batch_size=100)
    print('test loss, test acc:', results)
   
def _model(params): #ts_network
    lstm_output_size = 100
    filters,kernel_size,pool_size=32,8,4
    input_shape=(params['ts_len'], params['n_feats'])
    left_input = Input(input_shape)
    model = Sequential()
    model.add(Conv1D(filters,
                 kernel_size,
                 input_shape=input_shape,
                 padding='valid',
                 activation='relu',
                 strides=1))
    model.add(MaxPooling1D(pool_size))
    model.add(LSTM(lstm_output_size,dropout=0.5,kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(params['n_cats'],activation='sigmoid',name='hidden'))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.summary()
    return model

def cnn_model(params):
    input_layer = Input(shape=(params['ts_len'], params['n_feats'],1))
    activ='relu' #'elu'
    
    pool1=add_conv_layer(input_layer,0,activ=activ)#,pool_size=(,1))
    pool2=add_conv_layer(pool1,1,activ=activ)

    conv1=Conv2D(16, kernel_size=(8,1),
            activation=activ,name='conv1')(input_layer)
    pool1=MaxPooling2D(pool_size=(4,1),name='pool1')(conv1)

    conv2=Conv2D(16, kernel_size=(8,1),
            activation=activ,name='conv2')(pool1)
    pool2=MaxPooling2D(pool_size=(2,1),name='pool2')(conv2)

    pool2=Conv2D(16, kernel_size=(8,100),
            activation=activ,name='conv3')(pool2)
      

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
#    raise Exception("OK")
    return model

def lstm_model(params):
    ts_len,n_feats=(params['ts_len'], params['n_feats'])
    model_m = Sequential()
    model_m.add(Conv1D(64, 16, activation='relu', input_shape=(ts_len, n_feats)))
    model_m.add(Conv1D(64, 16, activation='relu'))
    model_m.add(MaxPooling1D(3))
    model_m.add(Conv1D(32, 8, activation='relu'))
    model_m.add(Conv1D(32, 8, activation='relu'))
    model_m.add(GlobalAveragePooling1D())
#    model_m.add(Flatten())
    model_m.add(Dense(64))
    model_m.add(Dropout(0.5))
    model_m.add(Dense(params['n_cats'], activation='softmax'))
    model_m.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True),
             metrics=['accuracy'])
    print(model_m.summary())
    return model_m

make_model("../auto/spline",out_path="../auto/lstm_nn",n_epochs=300)
#extract_features("../auto/spline","../auto/lstm_nn","../auto/lstm_feats")