import numpy as np
import keras,keras.utils
from sklearn.metrics import classification_report
#from keras.layers.normalization import BatchNormalization
import data,models


def simple_exp(in_path,out_path=None,n_epochs=1000):
    (X_train,y_train),(X_test,y_test)=data.make_dataset(in_path)
    n_cats,n_channels=data.get_params(X_train,y_train)
    X_train,y_train=prepare_data(X_train,y_train,n_channels)
    X_test,y_test=prepare_data(X_test,y_test,n_channels)
    model=models.make_res(n_cats,n_channels)
    model.fit(X_train,y_train,epochs=n_epochs,batch_size=256)
    test_model(X_test,y_test,model)
    if(out_path):
         model.save(out_path)

def test_model(X_test,y_test,model):
    raw_pred=model.predict(X_test,batch_size=256)
    pred_y,test_y=np.argmax(raw_pred,axis=1),np.argmax(y_test,axis=1)
    print(classification_report(test_y, pred_y,digits=4))

def prepare_data(X,y,n_channels):
    X=data.format_frames(X,n_channels)
    y=keras.utils.to_categorical(y)
    return X,y

#def train_binary_model(in_path,out_path,n_epochs=1500):
#    train_X,train_y=data.frame_dataset(in_path)
#    train_y=keras.utils.to_categorical(train_y)
#    files.make_dir(out_path)
#    n_cats=train_y.shape[1]
#    for cat_i in range(n_cats):
#        y_i=binarize(train_y,cat_i)
#        model=make_res(2)
#        model.summary()
#        model.fit(train_X,y_i,epochs=n_epochs,batch_size=256)
#        out_i=out_path+'/nn'+str(cat_i)
#        model.save(out_i)

#def binarize(train_y,cat_i):
#    y=np.zeros((train_y.shape[0],2))
#    for i,one_hot_i in enumerate(train_y):
#      j=int(one_hot_i[cat_i]==1)
#      y[i][j]=1 
#    return y