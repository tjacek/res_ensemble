import keras
from keras.models import Model,Sequential
from keras.layers import Input,Add,Dense, Dropout, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D, Activation
from keras.layers.merge import add
from keras import regularizers
from keras.initializers import glorot_uniform

import models.old

def get_model_factory(model_type):
    if(model_type=="old"):
        return models.old.make_conv,None
    if(model_type=="old_batch"):
        return models.old.make_conv,{'batch':True}
    if(model_type=="old_small"):
        return models.old.make_conv,{'hidden':64}
    return make_exp,None

def make_exp(n_cats,n_channels,params=None):
    X_input = Input(shape=(64,64,n_channels))
    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [16, 16, 32], stage = 1, block='a', s = 2)
    X = convolutional_block(X, f = 3, filters = [16, 16, 32], stage = 2, block='a', s = 2)

    reg=regularizers.l1(0.01)
    dense1 = Dense(100, activation='relu',name='hidden',kernel_regularizer=reg)(Flatten()(X))   

    drop1=Dropout(0.5)(dense1)
    output_layer = Dense(units=n_cats, activation='softmax')(drop1)
    model=Model(inputs=X_input, outputs=output_layer)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
 
              #optimizer=keras.optimizers.Adadelta())
    model.summary()
    return model

def convolutional_block(X, f, filters, stage, block, s=2):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def make_res(n_cats,n_channels):
    input_layer = Input(shape=(64,64,n_channels))
    activ='elu' #'elu'
    
    ker_reg=None#regularizers.l1(0.01)# if(l1) else None

    conv1=Conv2D(16, kernel_size=(5,5),
            activation=activ)(input_layer)
    pool1=MaxPooling2D(pool_size=(4,4))(conv1)
#    pool1=BatchNormalization()(pool1)

    conv2=Conv2D(16, kernel_size=(5,5),
            activation=activ)(pool1)
    pool2=MaxPooling2D(pool_size=(4,4))(conv1)
#    pool2=BatchNormalization()(pool2)

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

def make_exp1(n_cats,n_channels):
    model = Sequential()
    model.add(Conv2D(24, kernel_size=(5, 5),
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

def make_exp2(n_cats,n_channels):
    input_layer = Input(shape=(64,64,n_channels))
    activ='relu' #'elu'
    
    ker_reg=None#regularizers.l1(0.01)# if(l1) else None

    conv1=Conv2D(24, kernel_size=(5,5),
            activation=activ)(input_layer)
    pool1=MaxPooling2D(pool_size=(4,4))(conv1)

    conv2=Conv2D(16, kernel_size=(5,5),
            activation=activ)(pool1)
    pool2=MaxPooling2D(pool_size=(4,4))(conv1)

    n_hidden=32
    

    short1 = Dense(64, activation=activ)(Flatten()(pool2))   
    dense_layer1=Dense(64, activation=activ)(short1)
    res_layer1=add([short1, dense_layer1])
 
    short2 = Dense(n_hidden, activation=activ)(res_layer1)   
    dense_layer2=Dense(n_hidden, activation=activ)(short2)
    res_layer2=add([short2, dense_layer2],name='hidden')

    drop1=Dropout(0.5)(res_layer2)
    output_layer = Dense(units=n_cats, activation='softmax')(drop1)
    model=Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta())
    model.summary()
    return model
