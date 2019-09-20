from imgaug import augmenters as iaa
import imgs,data

def agum_simple(raw_path,agum_path):
    agum = iaa.Sequential([iaa.Affine(scale=(1.0, 1.5))])
    agum_template(raw_path,agum_path,agum,n_iters=1)


def agum_data(raw_path,agum_path):
    crop=iaa.CropAndPad(percent=(-0.15, 0.15),sample_independently=False,keep_size=False)
    agum =  iaa.Sequential([crop]) #iaa.SomeOf(2, [cro,crop_y])
    agum_template(raw_path,agum_path,agum,n_iters=2)


def agum_template(raw_path,agum_path,agum,n_iters=10):
    raw_data=imgs.read_seqs(raw_path)
    train,test=data.split(raw_data.keys())
    train_data={ name_i:raw_data[name_i] for name_i in train}
    agum_dict={}
    for name_i,seq_i in list(train_data.items()):
        agum_seq_i = agum(images=seq_i)
        for j in range(n_iters):
            new_name_i=name_i+'_'+str(j)
            print(new_name_i)
            agum_dict[new_name_i]=agum_seq_i
    new_dict={**raw_data,**agum_dict}
    imgs.save_seqs(new_dict,agum_path)