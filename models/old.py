import keras
from keras.models import Model,Sequential
from keras.layers import Input,Add,Dense, Dropout, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D, Activation
from keras import regularizers
from keras.models import load_model

def make_conv(n_cats,n_channels,params=None):
    if(not params):
        params={}
    n_hidden=params['hidden'] if('hidden' in params) else 100
    n_kerns1=params['n_kerns1'] if('n_kerns1' in params) else 16
    model = Sequential()
    model.add(Conv2D(n_kerns1, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(64,64,n_channels)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu',input_shape=(64,64,4)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(n_hidden, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01),))
    if(params and ('batch' in params)):
        model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
    return model

def make_pyramid(n_cats,n_channels,params=None):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(64,64,n_channels)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',input_shape=(64,64,4)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu',input_shape=(64,64,4)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(80, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01),))
    if(params and ('batch' in params)):
        model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(units=n_cats, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
    return model

def make_five(n_cats,n_channels,params=None):
    if(not params):
        params={}
    input_img = Input(shape=(64, 64, n_channels))
    x=input_img
    kern_size,pool_size,filters=(3,3),(2,2),[32,16,16,16]
    for filtr_i in filters:
        x = Conv2D(filtr_i, kern_size, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size, padding='same')(x)
    x=Flatten()(x)
    x=Dense(100, activation='relu',name="hidden",kernel_regularizer=regularizers.l1(0.01),)(x)
    x=Dropout(0.5)(x)
    x=Dense(units=n_cats,activation='softmax')(x)
    model = Model(input_img, x)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True))
    if(params['pretrain']):
        pretrain_model(model,params['pretrain'])
    return model 

def pretrain_model(model,auto_path):
    ae=load_model(auto_path)
    for i,layers_i in enumerate(ae.layers):
        weights_i=layers_i.get_weights()
        model.layers[i].set_weights(weights_i)
    return model