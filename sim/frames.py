import numpy as np,cv2
import keras
import keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input,Add,Dense, Dropout, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D,Activation,Lambda
import data,sim.gen,files

def show_frames(in_path,out_path):
    X_train,y_train=data.seq_dataset(in_path)
    X,y=sim.gen.gen_data(X_train,y_train)
    files.make_dir(out_path)
    for i,y_i in enumerate(y):
        x0,x1=X[i]
        cat_i=np.argmax(y_i)
        img_i=np.concatenate([x0,x1])
        out_i='%s/%d_%d.png' % (out_path,i,cat_i)
        print(out_i)
        cv2.imwrite(out_i,img_i)

def make_model(in_path):
    X_train,y_train=data.seq_dataset(in_path)
    X,y=sim.gen.gen_data(X_train,y_train)
    n_cats=data.count_cats(y)
    n_channels=int(X.shape[-2]/X.shape[-1])
    print(n_cats,n_channels)
    model=make_five(n_cats,n_channels,params=None)

def make_five(n_cats,n_channels,params=None):
    if(not params):
        params={}
    input_shape=(64,64,n_channels)
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    activ='relu'
    model.add(Conv2D(16, kernel_size=(4,4),activation=activ,name='conv1'))
    model.add(MaxPooling2D(pool_size=(2,2),name='pool1'))
    model.add(Conv2D(16, kernel_size=(4,4),activation=activ,name='conv2'))
    model.add(MaxPooling2D(pool_size=(2,2),name='pool2'))
    model.add(Conv2D(16, kernel_size=(4,4),activation=activ,name='conv3'))
    model.add(MaxPooling2D(pool_size=(2,2),name='pool3'))
    model.add(Flatten())
    model.add(Dense(64, activation=activ,name='hidden'))

    encoded_l = model(left_input)
    encoded_r = model(left_input)


    L2_layer = Lambda(lambda tensors:K.square(tensors[0] - tensors[1]))
    L2_distance = L1_layer([encoded_l, encoded_r])

    prediction = Dense(2,activation='sigmoid')(L2_distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    siamese_net.summary()
    optimizer = keras.optimizers.Adam(lr = 0.00006)
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    extractor=Model(inputs=model.get_input_at(0),outputs=model.get_layer("hidden").output)
    extractor.summary()
    return siamese_net,extractor
    return siamese_net