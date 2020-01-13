import numpy as np
import keras
import files
import sim,sim.gen,sim.dist
from itertools import product
import resnet
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten,GlobalAveragePooling1D
from keras.layers import Conv2D,Conv1D, MaxPooling1D,MaxPooling2D,Lambda,BatchNormalization

def extract(frame_path,model_path,out_path):
    files.make_dir(out_path)
    for i,model_i in enumerate(files.top_files(model_path)):
        frame_i="%s/nn%d" % (frame_path,i)
        out_i=out_path+"/nn"+str(i)
        sim.extract(frame_i,model_i,out_i)

def make_ens(in_path,out_path,n_epochs=10):
    files.make_dir(out_path)
    for i,in_i in enumerate(files.top_files(in_path)):
        (X_train,y_train),test,params=resnet.load_data(in_i,split=True)
        y_train=np.argmax(y_train,1)
        X_train=np.squeeze(X_train)
        gen_i=BinaryCats(i)
        X_i,y_i=gen_i(X_train,y_train)
        sim_metric,model=siamese_model(params)
        sim_metric.fit(X_i,y_i,epochs=n_epochs,batch_size=100)
#        raise Exception("OK")
        model.save(out_path+'/nn'+str(i))

class BinaryCats(object):
    def __init__(self,cat_i):
        self.cat_i=cat_i

    def __call__(self,X_old,y_old):#ts
        dist=sim.dist.make_balanced(y_old)
        in_i,out_i=dist.divide(self.cat_i) 
        pairs= list(product(in_i,in_i))
        pairs+= list(product(out_i,in_i))
        X,y=[],[]
        for pair_k in pairs:
            i,j=pair_k
            x_i,y_i=X_old[i],y_old[i] 
            x_j,y_j=X_old[j],y_old[j] 
            X.append((x_j,x_i))
            y.append(int(y_i==y_j))
        X,y=np.array(X),np.array(y)#keras.utils.to_categorical(y)
        X=[X[:,0],X[:,1]]
        return X,y

def siamese_model(params): #ts_network
    input_shape=(params['ts_len'], params['n_feats'])
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    add_mean(model)

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    prediction,loss=sim.good_loss(encoded_l,encoded_r)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    optimizer = keras.optimizers.Adam(lr = 0.00006)
    siamese_net.compile(loss=loss,optimizer=optimizer)
    extractor=Model(inputs=model.get_input_at(0),outputs=model.get_layer("hidden").output)
    extractor.summary()
    siamese_net.summary()
    return siamese_net,extractor

def add_mean(model):
    activ='relu'
    model.add(Conv1D(64, kernel_size=8,activation=activ,name='conv1'))
    model.add(MaxPooling1D(pool_size=4,name='pool1'))
    model.add(Conv1D(64, kernel_size=4,activation=activ,name='conv2'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation=activ))#,kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(16,activation=activ,name='hidden'))
    return model