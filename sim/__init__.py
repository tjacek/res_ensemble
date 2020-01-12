import keras,keras.backend as K
import numpy as np
from keras.models import load_model
#from keras.models import Model
import resnet,models.ts
import ens,local,files
from extract import save_seqs
import sim.gen
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten,GlobalAveragePooling1D
from keras.layers import Conv2D,Conv1D, MaxPooling1D,MaxPooling2D,Lambda
from keras import regularizers

def extract(frame_path,model_path,out_path=None):
    extractor=load_model(model_path)
    (X,y),names=resnet.load_data(frame_path,split=False)
    X=np.squeeze(X)
    X_feats=extractor.predict(X)
    resnet.get_feat_dict(X_feats,names,out_path)

def make_model(in_path,out_path=None,n_epochs=50):
    (X_train,y_train),test,params=resnet.load_data(in_path,split=True)
    X_train=np.squeeze(X_train)
    X,y=sim.gen.full_data(X_train,y_train)
#    make_models=models.ts.get_model_factory("sim_exp")
    sim_metric,model=siamese_model(params)
#    raise Exception("OK")
    sim_metric.fit(X,y,epochs=n_epochs,batch_size=100)
    if(out_path):
        model.save(out_path)

def siamese_model(params): #ts_network
    input_shape=(params['ts_len'], params['n_feats'])
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    add_mean(model)

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    prediction,loss=basic_loss(encoded_l,encoded_r)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    optimizer = keras.optimizers.Adam(lr = 0.00006)
    siamese_net.compile(loss=loss,optimizer=optimizer)
    extractor=Model(inputs=model.get_input_at(0),outputs=model.get_layer("hidden").output)
    extractor.summary()
    return siamese_net,extractor

def add_mean(model):
    activ='relu'
    model.add(Conv1D(64, kernel_size=4,activation=activ,name='conv1'))
    model.add(MaxPooling1D(pool_size=2,name='pool1'))
    model.add(Conv1D(32, kernel_size=4,activation=activ,name='conv2'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation=activ,name='hidden'))#,kernel_regularizer=regularizers.l1(0.01)))
    return model

def add_basic(model):
    activ='relu'
    model.add(Conv1D(64, kernel_size=8,activation=activ,name='conv1'))
    model.add(MaxPooling1D(pool_size=4,name='pool1'))
    model.add(Conv1D(32, kernel_size=8,activation=activ,name='conv2'))
    model.add(MaxPooling1D(pool_size=4,name='pool2'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation=activ,name='hidden'))#,kernel_regularizer=regularizers.l1(0.01)))
    return model

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

#def preproc_data(in_path,out_path,new_size=36,single=False):
#    def helper(in_i,out_i):
#        seqs_i=resnet.read_local_feats(in_i)
#        seqs_i={name_j:local.upsampling(seq_ij,new_size) 
#                 for name_j,seq_ij in seqs_i.items()}
#        save_seqs(seqs_i,out_i)
#    if(single):
#        helper(in_path,out_path)
#    else:
#        ens.template(in_path,out_path,helper)

#def make_ensemble(in_path,model_path,out_path,n_epochs=50):
#    files.make_dir(out_path)
#    def helper(in_i,out_i):
#        make_model(in_i,out_i,n_epochs)
#        feat_i=out_path+"/"+out_i.split("/")[-1]
#        extract(in_i,out_i,feat_i)
#    ens.template(in_path,model_path,helper)