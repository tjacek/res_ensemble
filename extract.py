import numpy as np
from keras.models import Model
from keras.models import load_model
import data,files

def extract_features(frame_path,model_path):
    seq_dict=data.dict_dataset(frame_path)
    models=read_models(model_path)
    extractors=[make_extractor(model_i) for model_i in models]
    frame_features(seq_dict,extractors[0])

def frame_features(data_dict,extractor):
    new_dict={}
    for name_i,seq_i in data_dict.items():
        new_dict[name_i]=extractor.predict(np.array(seq_i))
#        np.array([extractor.predict(frame_ij) for frame_ij in seq_i])
        print(new_dict[name_i].shape)
    return new_dict


def read_models(in_path):
    return [load_model(path_i) for path_i in files.top_files(in_path)]

def make_extractor(model):
    return Model(inputs=model.input,
                outputs=model.get_layer("hidden").output)