import keras
from keras.models import Model,Sequential
from keras.layers import Input,Add,Dense, Dropout, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D, Activation
from keras import regularizers
    
def make_conv(n_cats,n_channels,params=None):
    n_hidden=params['hidden'] if(params and ('hidden' in params)) else 100
    n_kerns1=params['n_kerns1'] if(params and ('n_kerns1' in params)) else 16
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
#              optimizer=keras.optimizers.Adadelta())
    return model
