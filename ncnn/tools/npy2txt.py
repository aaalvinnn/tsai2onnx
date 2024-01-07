# script for converting .npy to .txt. It is used for reading data in .cpp
import numpy as np
import os

def convert_npy2txt(dataset_path, txt_path):
    x = np.load(os.path.join(dataset_path, 'X_test.npy'))
    y = np.load(os.path.join(dataset_path, 'y_test.npy'))
    
    # x
    for i in range(len(y)):
        np.savetxt(os.path.join(txt_path, f'{i}.txt'), x[i], fmt='%d')

    # y
    np.savetxt(os.path.join(txt_path, 'labels.txt'), y, fmt='%d')

if __name__ == '__main__':
    # npy data path
    dataset_path = '../data/npydata_zmj'

    # store txt path
    txt_path = 'data'
    if not os.path.exists(txt_path):
        os.mkdir(txt_path)
    
    # convert and store
    convert_npy2txt(dataset_path, txt_path=txt_path)

    print('done')