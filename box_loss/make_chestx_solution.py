import csv
import os
import json

dir_path = '/home/yunjae_heo/SSD/yunjae.heo/chestx-det'
save_path = '/home/yunjae_heo/SSD/yunjae.heo/chestx-det'

train_anno_file = open(os.path.join(dir_path, "annotations", "chestx-det_train.json"), 'r')
train_anno = json.load(train_anno_file)
test_anno_file = open(os.path.join(dir_path, "annotations", "chestx-det_test.json"), 'r')
test_anno = json.load(test_anno_file)
new_train_file = open(os.path.join(save_path, "annotations", "chestx_multi_label_train.txt"),'w')
new_test_file = open(os.path.join(save_path, "annotations", "chestx_multi_label_test.txt"),'w')

anno_dict = dict()

tr_images = train_anno['images']
tr_labels = train_anno['annotations']

for anno in tr_labels:
    image_id = anno['image_id']
    image_name = tr_images[image_id-1]['file_name']
    label = anno['category_id']
    x1,y1,x2,y2 = anno['bbox']
    
    if image_id in anno_dict.keys():
        anno_dict[image_id].append([label,x1,y1,x2,y2])
    else:
        anno_dict[image_id] = [image_name]
        anno_dict[image_id].append([label,x1,y1,x2,y2])

for key in anno_dict.keys():
    line = anno_dict[key]
    for i in range(len(line)):
        if i == 0:
            new_train_file.write(f'{line[0]},')
        else:
            label,x1,y1,x2,y2 = line[i]
            new_train_file.write(f'{label} {x1} {y1} {x2} {y2},')
    new_train_file.write(f'\n')
new_train_file.close()

anno_dict = dict()

ts_images = test_anno['images']
ts_labels = test_anno['annotations']

for anno in ts_labels:
    image_id = anno['image_id']
    image_name = ts_images[image_id-1]['file_name']
    label = anno['category_id']
    x1,y1,x2,y2 = anno['bbox']
    
    if image_id in anno_dict.keys():
        anno_dict[image_id].append([label,x1,y1,x2,y2])
    else:
        anno_dict[image_id] = [image_name]
        anno_dict[image_id].append([label,x1,y1,x2,y2])

for key in anno_dict.keys():
    line = anno_dict[key]
    for i in range(len(line)):
        if i == 0:
            new_test_file.write(f'{line[0]},')
        else:
            label,x1,y1,x2,y2 = line[i]
            new_test_file.write(f'{label} {x1} {y1} {x2} {y2},')
    new_test_file.write(f'\n')
new_test_file.close()

train_anno_file.close()
test_anno_file.close()