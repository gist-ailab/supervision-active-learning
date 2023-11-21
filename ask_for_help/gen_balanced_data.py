import os, sys
from glob import glob
import shutil
import random

random.seed(0)

d_path = '/ailab_mat/dataset/ISIC_skin_disease/imageFolder'
s_path = '/ailab_mat/dataset/ISIC_skin_disease/balanced_train'
mode = 'test'
mask_path = '/ailab_mat/dataset/ISIC_skin_disease/Test_v2/ISIC-2017_Test_v2_Part1_GroundTruth'
mask_s_path = f'/ailab_mat/dataset/ISIC_skin_disease/balanced_train/mask_{mode}'
lbl_clss = ['bkl', 'nv', 'mel']

for clss in lbl_clss:
    root_path = os.path.join(d_path, mode, clss)
    save_path = os.path.join(s_path,mode,clss)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(os.path.join(mask_s_path, clss)):
        os.mkdir(os.path.join(mask_s_path, clss))
    img_path = glob(os.path.join(root_path, '*'))
    random.shuffle(img_path)
    selected_imgs = img_path[:]
    print(len(selected_imgs))
    for img_path in selected_imgs:
        img_name = img_path.split('/')[-1]
        shutil.copy(img_path, os.path.join(save_path, img_name))
        shutil.copy(os.path.join(mask_path, img_name.split('.')[0]+'_segmentation.png'), os.path.join(mask_s_path, clss, img_name.split('.')[0]+'_segmentation.png'))
    print("Done")
