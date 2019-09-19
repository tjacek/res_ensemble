import numpy as np
import keras,keras.utils
from sklearn.metrics import classification_report
#from keras.layers.normalization import BatchNormalization
import data,models


def simple_exp(in_path,out_path=None,n_epochs=500,model_type="conv"):
    (X_train,y_train),(X_test,y_test)=data.make_dataset(in_path)
    n_cats,n_channels=data.get_params(X_train,y_train)
    X_train,y_train=prepare_data(X_train,y_train,n_channels)
    X_test,y_test=prepare_data(X_test,y_test,n_channels) 
    model=make_model(model_type)(n_cats,n_channels)
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

def make_model(model_type):
    if(model_type=="res"):
        return models.make_res
    return models.make_conv