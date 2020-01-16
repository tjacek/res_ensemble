import numpy as np,cv2
import files,data,sim.frames


def sim_model(in_path,out_path):
    full=img_dataset(in_path)
    train,test=data.split(full.keys())
#    train={ name_i:full[name_i] for name_i in train}
    X0,X1,y=[],[],[]
    for i,name_i in enumerate(train):
        for name_j in train[i:]:
            y_k=int(name_i.split("_")[0]==name_j.split("_")[0])
            X0.append( full[name_i])
            X1.append(full[name_j])
            y.append(y_k)
    X=[np.array(X0),np.array(X1)]
    sim_metric,model=sim.frames.make_five(20,1)
    sim_metric.fit(X,y,epochs=250,batch_size=100)
    if(out_path):
        model.save(out_path)

def img_dataset(in_path):
    img_dict={}
    for path_i in files.top_files(in_path):
        name_i=files.clean_str(path_i.split("/")[-1])
        img_i=cv2.imread(path_i, cv2.IMREAD_GRAYSCALE)
        img_i=np.expand_dims(img_i,-1)
        img_dict[name_i]=img_i
    print(img_dict.keys())
    return img_dict