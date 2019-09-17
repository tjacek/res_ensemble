import numpy as np
import data,basic,files,models,extract

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

def binary_extract(frame_path,model_path,seq_path):
    files.make_dir(seq_path)
    for i,in_i in enumerate(files.top_files(model_path)):
        print(i)
        out_i= seq_path+'/'+in_i.split('/')[-1]
        extract.extract_features(frame_path,in_i,out_i)

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