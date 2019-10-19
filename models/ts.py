import keras
import keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,Lambda
from sklearn.metrics import classification_report
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add
from keras.models import load_model

def get_model_factory(model_type):
    if(model_type=='sim'):
        return siamese_model
    return make_conv

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

def siamese_model(params):
    input_shape=(params['ts_len'], params['n_feats'],1)
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    activ='relu'
    model.add(Conv2D(16, kernel_size=(8,1),activation=activ,name='conv1'))
    model.add(MaxPooling2D(pool_size=(4,1),name='pool1'))
    model.add(Conv2D(16, kernel_size=(8,1),activation=activ,name='conv2'))
    model.add(MaxPooling2D(pool_size=(4,1),name='pool2'))
    model.add(Flatten())
    model.add(Dense(100, activation=activ,name='hidden'))

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = Dense(2,activation='sigmoid')(L1_distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    optimizer = keras.optimizers.Adam(lr = 0.00006)
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    return siamese_net