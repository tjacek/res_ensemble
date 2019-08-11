import numpy as np
import keras,keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report
from keras import regularizers
import data

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
    X=[norm_seq(x_i) for x_i in X]
    X= np.expand_dims(np.array(X),-1)
    y=keras.utils.to_categorical(y)
    return X,y

def make_conv(params):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(8, 1),
                 activation='relu',
                 input_shape=(params['ts_len'],params['n_feats'],1)))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Conv2D(8, kernel_size=(8, 1),
                 activation='relu',
                 input_shape=(params['ts_len'],params['n_feats'],1)))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(units=params['n_cats'], activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01,  momentum=0.9, nesterov=True))
    return model

def norm_seq(x_i):
    mean_i=np.mean(x_i,axis=0)
    std_i=np.std(x_i,axis=0)
    def norm_helper(j,feature_j):
        feature_j-=mean_i[j]
        if(std_i[j]!=0):
            feature_j/=std_i[j]
        return feature_j
    return np.array([norm_helper(j,feat_j) 
                        for j,feat_j in enumerate(x_i.T)]).T  


train_conv("../imgset")