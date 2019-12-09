def make_basic(n_channels):
    input_img = Input(shape=(64, 64, n_channels))
    x=input_img
    kern_size,pool_size,filters=(3,3),(2,2),[32,16,16,16]
    for filtr_i in filters:
        x = Conv2D(filtr_i, kern_size, activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size, padding='same')(x)
    shape = K.int_shape(x)
    x=Flatten()(x)
    encoded=Dense(100)(x) 
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
