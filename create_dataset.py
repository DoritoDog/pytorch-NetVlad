from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import os
import shutil
import argparse

parser = argparse.ArgumentParser(description='Create a dataset containing specified images.' +
'Copies an existing MATLAB NetVLAD dataset file.')
parser.add_argument('--source', type=str, default='pitts30k_train')
parser.add_argument('--database_images_path', type=str, default='/content/pytorch-NetVlad/000')
parser.add_argument('--query_images_path', type=str, default='/content/pytorch-NetVlad/queries_real/000')
parser.add_argument('--result_database_size', type=int, default=10)
parser.add_argument('--result_queries_size', type=int, default=10)
opt = parser.parse_args()

datasets_path = '/content/pytorch-NetVlad/datasets/'
source_file_name = opt.source + '_original.mat'
created_file_name = opt.source + '.mat'
source_file_path = os.path.join(datasets_path, source_file_name)
created_file_path = os.path.join(datasets_path, created_file_name)

if not os.path.exists(source_file_path):
  shutil.copyfile(created_file_path, source_file_path)
mat = loadmat(created_file_path)

def setDbStructImages(isDatabase):
  dbStruct_images_index = 1 if isDatabase else 3
  # Resizes the dbStruct images.
  size = opt.result_database_size if isDatabase else opt.result_queries_size
  mat['dbStruct'][0][0][dbStruct_images_index] = mat['dbStruct'][0][0][dbStruct_images_index][:size]

  images_dir = opt.database_images_path if isDatabase else opt.query_images_path
  image_file_names = os.listdir(images_dir)
  for i in range(1, num + 1):
    image_file_name = os.path.join(images_dir, image_file_names[i])

    # '<U' denotes a little endian string.
    dtype = '<U' + (str)(len(image_file_name))
    mat['dbStruct'][0][0][dbStruct_images_index][i-1][0] = np.array([image_file_name]).astype(dtype)

mat['dbStruct'][0][0][5][0][0] = opt.result_database_size
mat['dbStruct'][0][0][6][0][0] = opt.result_queries_size

mat['dbStruct'][0][0][2] = np.ones((2, 10))
mat['dbStruct'][0][0][4] = np.ones((2, 10))

setImages(True)
setImages(False)
savemat('/content/pytorch-NetVlad/datasets/pitts30k_train.mat', mat)
savemat('/content/pytorch-NetVlad/datasets/pitts30k_val.mat', mat)
print(mat)
