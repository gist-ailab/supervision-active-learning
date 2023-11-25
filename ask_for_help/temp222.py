import os, sys
from glob import glob
import shutil
import json

modes = ['train', 'val', 'test']

for mode in modes:
    img_path = '/ailab_mat/dataset/amphi/images/'
    save_path = f'/ailab_mat/dataset/amphi/{mode}/'
    anno_path = f'/ailab_mat/dataset/amphi/annotations/{mode}1.json'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(anno_path, 'r') as f:
        annotations = json.load(f)
    img_annos = annotations['images']

    print(len(img_annos))
    for anno in img_annos:
        file_name = os.path.join(img_path, anno['file_name'])
        shutil.copyfile(file_name, os.path.join(save_path, anno['file_name']))