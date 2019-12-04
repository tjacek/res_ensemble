import numpy as np
import os,keras
from keras.layers import Input, Dense,Conv2D,Reshape,Conv2DTranspose
from keras.layers import Flatten,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.models import load_model
import data,imgs

def train(in_path,out_path=None,n_epochs=20,recon=True):
    (X_train,y_train),(X_test,y_test)=data.make_dataset(in_path)
    n_cats,n_channels=data.get_params(X_train,y_train)
    X=data.format_frames(X_train,n_channels)
    decode,encode=make_basic(n_channels)
    model=decode if(recon) else decode
    model.summary()
    model.fit(X,X,epochs=n_epochs,batch_size=256)#,
#    	shuffle=True,validation_data=(X, X))
    if(not out_path):
        dest_dir=os.path.split(in_path)[0]
        out_path=dest_dir+'/ae'
    print(out_path)
    model.save(out_path)

def reconstruct(in_path,model_path,out_path=None,diff=False):
    model=load_model(model_path)
    if(not out_path):
        out_path=os.path.split(in_path)[0]+'/rec'	
    def rec_helper(X):
        X=np.array(X)
        X=data.format_frames(X)
        pred= model.predict(X)
        if(diff):            	
            pred=np.abs(pred-X)
        pred=  [np.vstack(frame_i.T) for frame_i in pred]
        return pred   
    imgs.transform(in_path,out_path,rec_helper,False)

def make_autoencoder(n_channels):
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

def make_basic(n_channels):
    input_img = Input(shape=(64, 64, n_channels))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    shape = K.int_shape(x)
    x=Flatten()(x)
    encoded=Dense(100)(x) 
    x = Dense(shape[1]*shape[2]*shape[3])(encoded)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(16, (3, 3), activation='relu',padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(16, (3, 3), activation='relu',padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(16, (3, 3), activation='relu',padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(32, (3, 3), activation='relu',padding='same')(x)
    x=Conv2DTranspose(filters=2,kernel_size=64,padding='same')(x)
    model = Model(input_img, x)
    model.compile(optimizer='adam',
    	                loss='mean_squared_error')
    return model,None

#train('../time/data' )
reconstruct("../time/data","../time/ae")