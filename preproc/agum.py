from imgaug import augmenters as iaa
import imgs,data

def agum_data(raw_path,agum_path):
    raw_data=imgs.read_seqs(raw_path)
    train,test=data.split(raw_data.keys())
    train_data={ name_i:raw_data[name_i] for name_i in train}
    agum = iaa.Sequential([iaa.Crop(px=(0, 32))])
    agum_dict={}
    for name_i,seq_i in list(train_data.items()):
        agum_seq_i = agum(images=seq_i)
        new_name_i=name_i+'_0'
        print(new_name_i)
        agum_dict[new_name_i]=agum_seq_i
    new_dict={**raw_data,**agum_dict}
    imgs.save_seqs(new_dict,agum_path)