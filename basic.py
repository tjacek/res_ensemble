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
        model=make_conv(2)
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

def binarize(train_y,cat_i):
    y=np.zeros((train_y.shape[0],2))
    for i,one_hot_i in enumerate(train_y):
      j=int(one_hot_i[cat_i]==1)
      y[i][j]=1 
    return y