import numpy as np
import h5py
from DirectoryEmbedded import get_files, DirectoryEmbedded

label_size = 5
image_size = 95

def read_folder(directory):
    data = None
    data_batch = None
    labels = None
    info = []
    index = 0
    
    for file_name, info_name in zip(get_files(directory, 'images'), get_files(directory, 'info')):
        print('Opened file successfully')
        with open(file_name, 'r') as f, open(info_name, 'r') as g:
            for line, info_line in zip(f, g):
                index += 1
                
                py_list = list(map(float, line[:-1].split(', ')))
                info.append(info_line[:-1])
                
                new_data = np.array([py_list[:-label_size**2]]) # Make integer later after image collapse, dtype=np.int32)
                new_labels = np.array([py_list[-label_size**2:]]) # Make integer later after image collapse, dtype=np.int32)
                
                if data_batch is None:
                    data_batch = new_data
                elif not data_batch.shape[0] % 50:
                    if data is None:
                        data = data_batch
                    else:
                        data = np.vstack((data, data_batch))
                    data_batch = new_data
                else:
                    data_batch = np.vstack((data_batch, new_data))
                    
                if labels is not None:
                    labels = np.vstack((labels, new_labels))
                else:
                    labels = new_labels
                
                if not index % 50:
                    print(index)
    
    #Wrap Up
    data = np.reshape(np.vstack((data, data_batch)), [-1, image_size, image_size, 1])
    labels = np.reshape(labels, [-1, label_size, label_size])
    info = np.array(info, dtype=np.string_)
    
    print(data.shape)
    return data, labels, info

def write_folder(directory, images, labels, info):
    f = h5py.File(DirectoryEmbedded(directory).file('data.h5py'), 'w')
    
    f.create_dataset('images', data=images, dtype='i')
    f.create_dataset('labels', data=labels, dtype='i')
    f.create_dataset('info', data=info)
    
    f.close()

def open_file(file_name):
    f = h5py.File(file_name, 'r')
    
    return f['images'][...], f['labels'][...], f['info'][...]
