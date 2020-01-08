import resnet
from keras.models import Sequential
from keras.layers import Input,LSTM,Dense,Activation
from keras.layers.convolutional import Conv3D,MaxPooling1D,Conv1D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np


def make_model(in_path,out_path=None,n_epochs=50):
    (X_train,y_train),(X_test,y_test),params=resnet.load_data(in_path,split=True)
    X_train,X_test=np.squeeze(X_train),np.squeeze(X_test)
#    raise Exception(X_train.shape)
    model=lstm_model(params)
    model.fit(X_train,y_train,epochs=n_epochs,batch_size=100)
    if(out_path):
        model.save(out_path)
    results = model.evaluate(X_test, y_test, batch_size=100)
    print('test loss, test acc:', results)

def lstm_model(params): #ts_network
    lstm_output_size = 64
    filters,kernel_size,pool_size=16,8,4
    input_shape=(params['ts_len'], params['n_feats'])
    left_input = Input(input_shape)
    model = Sequential()
    model.add(Conv1D(filters,
                 kernel_size,
                 input_shape=input_shape,
                 padding='valid',
                 activation='relu',
                 strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(params['n_cats']))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.summary()
    return model

make_model("../auto/spline",out_path=None,n_epochs=50)