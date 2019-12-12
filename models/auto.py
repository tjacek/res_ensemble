from keras.layers import Input, Dense,Conv2D,Reshape,Conv2DTranspose
from keras.layers import Flatten,MaxPooling2D,UpSampling2D

def get_model_factory(model_type):
    if(model_type=='basic'):
        return make_basic,{}
     if(model_type=='pyr'):
        params={'n_hidden':64,'filters':[32,32,16,16]}
        return make_basic,{}
    return make_autoencoder,{}

def make_basic(n_channels,params):
    input_img = Input(shape=(64, 64, n_channels))
    x=input_img
    kern_size,pool_size=(3,3),(2,2)
    n_hidden= params['n_hidden'] if( 'n_hidden' in params) else 100
    filters= params['filters'] if('filters' in params) else [32,16,16,16]
    for filtr_i in filters:
        x = Conv2D(filtr_i, kern_size, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size, padding='same')(x)
    shape = K.int_shape(x)
    x=Flatten()(x)
    encoded=Dense(n_hidden)(x) 
    x = Dense(shape[1]*shape[2]*shape[3])(encoded)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    filters.reverse()
    for filtr_i in filters:
      x = UpSampling2D(pool_size)(x)
      x = Conv2DTranspose(filtr_i, kern_size, activation='relu',padding='same')(x)
    x=Conv2DTranspose(filters=2,kernel_size=64,padding='same')(x)
    recon = Model(input_img, x)
    model =Model(input_img,encoded)
    recon.compile(optimizer='adam',
                      loss='mean_squared_error')
    return model,recon

def make_autoencoder(n_channels,params):
    input_img = Input(shape=(64, 64, n_channels))
    n_kerns=32
    x = Conv2D(n_kerns, (5, 5), activation='relu',padding='same')(input_img)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu',padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    shape = K.int_shape(x)
    x=Flatten()(x)
    encoded=Dense(100)(x)    
    x = Dense(shape[1]*shape[2]*shape[3])(encoded)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(16, (5, 5), activation='relu',padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(n_kerns, (5, 5), activation='relu',padding='same')(x)
    
    x=Conv2DTranspose(filters=2,kernel_size=n_kerns,padding='same')(x)
    recon=Model(input_img,encoded)
    autoencoder = Model(input_img, x) 
    autoencoder.compile(optimizer='adam',#keras.optimizers.SGD(lr=0.0001,  momentum=0.9, nesterov=True), 
                      loss='mean_squared_error')
    return autoencoder,recon