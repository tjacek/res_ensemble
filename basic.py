import keras,keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
import data,files

def train_binary_model(in_path,out_path,n_epochs=1500):
    train_X,train_y=data.read_imgs(in_path)
    train_y=keras.utils.to_categorical(y)
    files.make_dir(out_path)
    n_cats=train_y.shape[1]
    for cat_i in range(n_cats):
        y_i=binarize(train_y,cat_i)
        model=make_res(2)
        model.summary()
        model.fit(train_X,y_i,epochs=n_epochs,batch_size=256)
        out_i=out_path+'/nn'+str(cat_i)
        model.save(out_i)

def make_conv(n_cats):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=(64,64,4)))
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
    input_layer = Input(shape=(64,64,1))
    activ='relu' #'elu'

    conv1=Conv2D(16, kernel_size=(5,5),
            activation=activ)(input_layer)
    pool1=MaxPooling2D(pool_size=(4,4))(conv1)
    
    conv2=Conv2D(16, kernel_size=(5,5),
            activation=activ)(pool1)
    pool2=MaxPooling2D(pool_size=(4,4))(conv1)
    
    n_hidden=100
    
    short1 = Dense(n_hidden, activation=activ)(Flatten()(pool2))   
    dense_layer1=Dense(n_hidden, activation=activ)(short1)
    res_layer1=add([short1, dense_layer1])

    short2 = Dense(n_hidden, activation=activ)(res_layer1)   
    dense_layer2=Dense(n_hidden, activation=activ)(short2)
    res_layer2=add([short2, dense_layer2])

    short3 = Dense(n_hidden, activation=activ)(res_layer2)   
    dense_layer3=Dense(n_hidden, activation=activ)(short3)
    res_layer3=add([short3, dense_layer3],name='hidden')

    drop1=Dropout(0.5)(res_layer3)
    output_layer = Dense(units=params['n_cats'], activation='softmax')(drop1)
    model=Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta())
    model.summary()
    return model

def binarize(train_y,cat_i):
    y=np.zeros((train_y.shape[0],2))
    for i,one_hot_i in enumerate(train_y):
      j=int(one_hot_i[cat_i]==1)
      y[i][j]=1 
    return y