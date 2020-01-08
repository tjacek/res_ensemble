import numpy as np,cv2
import keras
import keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input,Add,Dense, Dropout, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D,Activation,Lambda
from keras import regularizers
from keras.models import load_model
import data,sim.gen,files,imgs,extract

def show_frames(in_path,out_path):
    (X_train,y_train),test=data.make_dataset(in_path,False)
    X,y=sim.gen.OneCat(14)(X_train,y_train)#balanced_data(X_train,y_train)
    files.make_dir(out_path)
    for i,y_i in enumerate(y):
        x0,x1=X[0][i],X[1][i]
        cat_i=np.argmax(y_i)
        img_i=np.concatenate([x0,x1])
        out_i='%s/%d_%d.png' % (out_path,i,cat_i)
        print(out_i)
        cv2.imwrite(out_i,img_i)

def extract_feats(frame_path,model_path,out_path=None):
    extractor=load_model(model_path)
    img_seqs=imgs.read_seqs(frame_path)
    feats_seq={name_i:data.format_frames(seq_i)  
                for name_i,seq_i in img_seqs.items()}
    feat_dict={name_i:extractor.predict(seq_i) 
                for name_i,seq_i in feats_seq.items()}
    extract.save_seqs(feat_dict,out_path)

def sim_ens(in_path,out_path,n_epochs=500,n_cats=20):
    files.make_dir(out_path)
    for i in range(n_cats):
        out_i="%s/nn%d"%(out_path,i)
        make_model(in_path,out_i,n_epochs=n_epochs,gen_type=("one",i))

def make_model(in_path,out_path,n_epochs=500,gen_type="balanced"):
    (X_train,y_train),test=data.make_dataset(in_path,False)
    gen=sim.gen.get_data_generator(gen_type)
    X,y=gen(X_train,y_train)
    X,n_cats,n_channels=prepare_data(X,y)
#    raise Exception(n_cats)
    sim_metric,model=make_five(n_cats,n_channels,params=None)
    sim_metric.fit(X,y,epochs=n_epochs,batch_size=128)
    if(out_path):
        model.save(out_path)

def prepare_data(X,y):
    n_cats=data.count_cats(y)
    X= [data.format_frames(x_i) for x_i in X]
    n_channels=X[0].shape[-1]
    print(n_cats,n_channels)
    return X,n_cats,n_channels

def make_five_(n_cats,n_channels,params=None):
    if(not params):
        params={}
    input_shape=(64,64,n_channels)
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    activ='relu'
    model.add(Conv2D(64, kernel_size=(4,4),activation=activ,name='conv1'))
    model.add(MaxPooling2D(pool_size=(2,2),name='pool1'))
    model.add(Conv2D(32, kernel_size=(4,4),activation=activ,name='conv2'))
    model.add(MaxPooling2D(pool_size=(2,2),name='pool2'))
    model.add(Conv2D(16, kernel_size=(4,4),activation=activ,name='conv3'))
    model.add(MaxPooling2D(pool_size=(2,2),name='pool3'))
    model.add(Flatten())
    model.add(Dense(64, activation=activ,name='hidden',kernel_regularizer=regularizers.l1(0.01)))

    encoded_l = model(left_input)
    encoded_r = model(right_input)


    L2_layer = Lambda(lambda tensors:K.square(tensors[0] - tensors[1]))
    L2_distance = L2_layer([encoded_l, encoded_r])

    prediction = Dense(2,activation='sigmoid')(L2_distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    siamese_net.summary()
    optimizer = keras.optimizers.Adam(lr = 0.00006)#keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True)
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)
    extractor=Model(inputs=model.get_input_at(0),outputs=model.get_layer("hidden").output)
    return siamese_net,extractor


def make_five(n_cats,n_channels,params=None):
    if(not params):
        params={}
    input_shape=(64,64,n_channels)
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    activ='relu'
    kern_size,pool_size,filters=(3,3),(2,2),[32,16,16,16]
    for filtr_i in filters:
        model.add(Conv2D(filtr_i, kern_size, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size, padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation=activ,name='hidden',kernel_regularizer=regularizers.l1(0.01)))

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    prediction,loss=contr_loss(encoded_l,encoded_r)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    optimizer = keras.optimizers.Adam(lr = 0.00006)#keras.optimizers.SGD(lr=0.001,  momentum=0.9, nesterov=True)
    siamese_net.compile(loss=loss,#"binary_crossentropy",
        optimizer=optimizer)
    extractor=Model(inputs=model.get_input_at(0),outputs=model.get_layer("hidden").output)
    extractor.summary()
    return siamese_net,extractor

def basic_loss(encoded_l,encoded_r):
    L2_layer = Lambda(lambda tensors:K.square(tensors[0] - tensors[1]))
    L2_distance = L2_layer([encoded_l, encoded_r])
    return Dense(2,activation='sigmoid')(L2_distance),"binary_crossentropy"

def contr_loss(encoded_l,encoded_r):
    L2_layer = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)
    return L2_layer([encoded_l, encoded_r]),contrastive_loss

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)