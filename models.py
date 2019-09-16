import keras
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import add
from keras import regularizers

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