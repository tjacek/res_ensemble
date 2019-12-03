import os,re

def top_files(path):
    paths=[ path+'/'+file_i for file_i in os.listdir(path)]
    paths=sorted(paths,key=natural_keys)
    return paths

def bottom_files(path,full_paths=True):
    all_paths=[]
    for root, directories, filenames in os.walk(path):
        if(not directories):
            for filename_i in filenames:
                path_i= root+'/'+filename_i if(full_paths) else filename_i
                all_paths.append(path_i)
    all_paths.sort(key=natural_keys)        
    return all_paths

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def atoi(text):
    return int(text) if text.isdigit() else text

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def clean_str(name_i):
    digits=[ str(int(digit_i)) for digit_i in re.findall(r'\d+',name_i)]
    return "_".join(digits)