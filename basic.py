import numpy as np
import keras,keras.utils
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import add
from keras import regularizers
from sklearn.metrics import classification_report
#from keras.layers.normalization import BatchNormalization
import data#,files


def simple_exp(in_path,n_epochs=10):
    (X_train,y_train),(X_test,y_test)=data.make_dataset(in_path)
    n_cats=count_cats(y_train)
    n_channels=count_channels(X_train) 
    X_train,y_train=prepare_data(X_train,y_train,n_channels)
    X_test,y_test=prepare_data(X_test,y_test,n_channels)

    model=make_conv(n_cats,n_channels)
    model.fit(X_train,y_train,epochs=n_epochs,batch_size=256)

    raw_pred=model.predict(X_test,batch_size=256)
    pred_y,test_y=np.argmax(raw_pred,axis=1),np.argmax(y_test,axis=1)
    print(classification_report(test_y, pred_y,digits=4))

def prepare_data(X,y,n_channels):
    X=format_frames(X,n_channels)
    y=keras.utils.to_categorical(y)
    return X,y

def count_cats(y):
    return np.unique(np.array(y)).shape[0]

def count_channels(X):
    frame_dims=X[0].shape
    return int(frame_dims[0]/frame_dims[1])

def format_frames(frames ,n_channels):
    return np.array([np.array(np.vsplit(frame_i,n_channels)).T
                      for frame_i in frames])

#def train_binary_model(in_path,out_path,n_epochs=1500):
#    train_X,train_y=data.frame_dataset(in_path)
#    train_y=keras.utils.to_categorical(train_y)
#    files.make_dir(out_path)
#    n_cats=train_y.shape[1]
#    for cat_i in range(n_cats):
#        y_i=binarize(train_y,cat_i)
#        model=make_res(2)
#        model.summary()
#        model.fit(train_X,y_i,epochs=n_epochs,batch_size=256)
#        out_i=out_path+'/nn'+str(cat_i)
#        model.save(out_i)

def make_conv(n_cats,n_channels):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(64,64,n_channels)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu',input_shape=(64,64,4)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01),))
    model.add(Dropout(0.5))
    model.add(Dense(units=n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              #optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
              optimizer=keras.optimizers.Adadelta())
    return model

def make_res(n_cats):
    input_layer = Input(shape=(64,64,4))
    activ='elu' #'elu'
    
    ker_reg=None#regularizers.l1(0.01)# if(l1) else None

    conv1=Conv2D(16, kernel_size=(5,5),
            activation=activ)(input_layer)
    pool1=MaxPooling2D(pool_size=(4,4))(conv1)
    pool1=BatchNormalization()(pool1)

    conv2=Conv2D(16, kernel_size=(5,5),
            activation=activ)(pool1)
    pool2=MaxPooling2D(pool_size=(4,4))(conv1)
    pool2=BatchNormalization()(pool2)

    n_hidden=100
    
    short1 = Dense(n_hidden, activation=activ)(Flatten()(pool2))   
    dense_layer1=Dense(n_hidden, activation=activ)(short1)
    res_layer1=add([short1, dense_layer1])

    short2 = Dense(n_hidden, activation=activ)(res_layer1)   
    dense_layer2=Dense(n_hidden, activation=activ)(short2)
    res_layer2=add([short2, dense_layer2])

    short3 = Dense(n_hidden, activation=activ,kernel_regularizer=regularizers.l1(0.01))(res_layer2)   
    dense_layer3=Dense(n_hidden, activation=activ,kernel_regularizer=regularizers.l1(0.01))(short3)
    res_layer3=add([short3, dense_layer3],name='hidden',)

    drop1=Dropout(0.5)(res_layer3)
    output_layer = Dense(units=n_cats, activation='softmax')(drop1)
    model=Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta())
    model.summary()
    return model

#def binarize(train_y,cat_i):
#    y=np.zeros((train_y.shape[0],2))
#    for i,one_hot_i in enumerate(train_y):
#      j=int(one_hot_i[cat_i]==1)
#      y[i][j]=1 
#    return y