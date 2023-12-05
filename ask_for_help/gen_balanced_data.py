import os, sys
from glob import glob
import shutil
import random
import albumentations as A

random.seed(0)

d_path = '/ailab_mat/dataset/ISIC_skin_disease/imageFolder'
s_path = '/ailab_mat/dataset/ISIC_skin_disease/melanoma_classification'

if not os.path.isdir(s_path):
    os.mkdir(s_path)

mode = 'train'
mask_path = os.path.join(d_path, f'mask_{mode}')
mask_s_path = os.path.join(s_path, f'mask_{mode}')
lbl_clss = ['bkl', 'nv', 'mel']

if not os.path.isdir(os.path.join(s_path, mode)):
    os.mkdir(os.path.join(s_path, mode))

if not os.path.isdir(mask_s_path):
    os.mkdir(mask_s_path)

for clss in lbl_clss:
    if clss=='bkl': last=62
    if clss=='nv': last=314
    if clss=='mel': last=374

    root_path = os.path.join(d_path, mode, clss)
    save_path = os.path.join(s_path,mode,clss)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(os.path.join(mask_s_path, clss)):
        os.mkdir(os.path.join(mask_s_path, clss))
    img_path = glob(os.path.join(root_path, '*'))
    random.shuffle(img_path)
    selected_imgs = img_path[:last]
    print(len(selected_imgs))
    for img_path in selected_imgs:
        img_name = img_path.split('/')[-1]
        shutil.copy(img_path, os.path.join(save_path, img_name))
        shutil.copy(os.path.join(mask_path, clss, img_name.split('.')[0]+'_segmentation.png'), os.path.join(mask_s_path, clss, img_name.split('.')[0]+'_segmentation.png'))
    print("Done")
