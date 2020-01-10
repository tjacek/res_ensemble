import resnet
from keras.models import Model,Sequential
from keras.layers import Input,LSTM,Dense,Activation
from keras.layers.convolutional import Conv3D,MaxPooling1D,Conv1D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.models import load_model

def extract_features(frame_path,model_path,out_path):
    (X,y),names=resnet.load_data(frame_path,split=False)
    X=np.squeeze(X)
    model=load_model(model_path)
    extractor=Model(inputs=model.input,
                outputs=model.get_layer("hidden").output)
    X_feats=extractor.predict(X)
    return resnet.get_feat_dict(X_feats,names,out_path)

#def extract_features(frame_path,model_path,out_path):
#    (X_train,y_train),(X_test,y_test),params=resnet.load_data(frame_path,split=True)
#    X_test=np.squeeze(X_test)
#    model=load_model(model_path)
#    results = model.evaluate(X_test, y_test, batch_size=100)
#    y_pred=model.predict(X_test)
#    print([np.argmax(y_i) for y_i in y_test])
#    print([np.argmax(y_i) for y_i in y_pred])
#    print('test loss, test acc:', results)

def make_model(in_path,out_path=None,n_epochs=50):
    (X_train,y_train),(X_test,y_test),params=resnet.load_data(in_path,split=True)
    X_train,X_test=np.squeeze(X_train),np.squeeze(X_test)
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
    model.add(LSTM(lstm_output_size,dropout=0.5))
    model.add(Dense(params['n_cats'],activation='sigmoid',name='hidden'))
#    model.add(Activation('sigmoid'))

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.summary()
    return model

make_model("../auto/spline",out_path="../auto/lstm_nn",n_epochs=200)
#extract_features("../auto/spline","../auto/lstm_nn","../auto/lstm_feats")