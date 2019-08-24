import files,resnet,basic

def binary(in_path,out_path,n_epochs=10):
    (train_X,train_y),(test_X,test_y),params=resnet.load_data(in_path)
    files.make_dir(out_path)
    n_cats=train_y.shape[1]
    for i in range(n_cats):
        params['n_cats']=2
        model=resnet.make_conv(params)
        y_i= basic.binarize(train_y,i)
        model.fit(train_X,y_i,epochs=n_epochs,batch_size=100)
        out_i=out_path+'/nn'+str(i)
        model.save(out_i)