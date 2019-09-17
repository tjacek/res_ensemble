#from keras.models import Model,Sequential
#from keras.models import load_model
import numpy as np
import data,basic,files,models#,data,resnet,feats

def train_binary_model(in_path,out_path,n_epochs=1500):
    (X_train,y_train),(X_test,y_test)=data.make_dataset(in_path)
    n_cats,n_channels=data.get_params(X_train,y_train)
    X_train,y_train=basic.prepare_data(X_train,y_train,n_channels)
    X_test,y_test=basic.prepare_data(X_test,y_test,n_channels)
    files.make_dir(out_path)
    for cat_i in range(n_cats):        
        y_i=binarize(y_train,cat_i)
        model=models.make_conv(2,n_channels)
        model.summary()
        model.fit(X_train,y_i,epochs=n_epochs,batch_size=256)
        out_i=out_path+'/nn'+str(cat_i)
        model.save(out_i)

def binarize(train_y,cat_i):
    binary_y=[]
    for sample_j in train_y:
        new_sample_j=np.zeros((2,))
        if(sample_j[cat_i]==1):
            new_sample_j[0]=1
        else:
            new_sample_j[1]=1
        binary_y.append(new_sample_j)
    return np.array(binary_y)

#def binary(in_path,out_path,n_epochs=500):
#    (train_X,train_y),(test_X,test_y),params=resnet.load_data(in_path)
#    files.make_dir(out_path)
#    n_cats=train_y.shape[1]
#    for i in range(n_cats):
#        params['n_cats']=2
#        model=resnet.make_conv(params)
#        y_i= basic.binarize(train_y,i)
#        model.fit(train_X,y_i,epochs=n_epochs,batch_size=100)
#        out_i=out_path+'/nn'+str(i)
#        model.save(out_i)

#def binary_extract(in_path,frame_path,out_path,cat_feats=False):
#    model_paths=files.top_files(in_path)
#    models=[load_model(path_i) for path_i in model_paths]
#    if(cat_feats):
#        extractors=models	
#    else:
#        extractors=[get_extractor(model_i) for model_i in models]
#    X,y,names=data.img_dataset(frame_path,split_data=False)
#    X,y=resnet.prepare_data(X,y)
#    files.make_dir(out_path)
#    for i,extr_i in enumerate(extractors):
#        X_feats=extr_i.predict(X)
#        feat_dict={ names[i]:feat_i for i,feat_i in enumerate(X_feats)}
#        out_i=out_path+'/'+model_paths[i].split('/')[-1]
#        print(out_i)
#        feats.save_feats(feat_dict,out_i)

#def get_extractor(model):
#    return Model(inputs=model.input,
#                outputs=model.get_layer("hidden").output)