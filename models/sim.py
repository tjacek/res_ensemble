import keras
import keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,Lambda

def siamese_model(params):
    input_shape=(params['ts_len'], params['n_feats'],1)
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    activ='relu'
    model.add(Conv2D(16, kernel_size=(4,1),activation=activ,name='conv1'))
    model.add(MaxPooling2D(pool_size=(2,1),name='pool1'))
    model.add(Conv2D(16, kernel_size=(4,1),activation=activ,name='conv2'))
    model.add(MaxPooling2D(pool_size=(2,1),name='pool2'))
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
    extractor=Model(inputs=model.get_input_at(0),outputs=model.get_layer("hidden").output)
    extractor.summary()
    return siamese_net,extractor

def siamese_exp(params):
    input_shape=(params['ts_len'], params['n_feats'],1)
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    activ='relu'
    model.add(Conv2D(16, kernel_size=(4,1),activation=activ,name='conv1'))
    model.add(MaxPooling2D(pool_size=(2,1),name='pool1'))
    model.add(Conv2D(16, kernel_size=(4,1),activation=activ,name='conv2'))
    model.add(MaxPooling2D(pool_size=(4,1),name='pool2'))
    model.add(Flatten())
    model.add(Dense(64, activation=activ,name='hidden'))

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    L1_layer = Lambda(lambda tensors:K.square(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = Dense(2,activation='sigmoid')(L1_distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    optimizer = keras.optimizers.Adam(lr = 0.00006)
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    extractor=Model(inputs=model.get_input_at(0),outputs=model.get_layer("hidden").output)
    extractor.summary()
    return siamese_net,extractor

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def acc(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))