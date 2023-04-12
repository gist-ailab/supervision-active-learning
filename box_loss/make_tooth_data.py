import csv
import os
from glob import glob
import random

random.seed(20230404)

dir_path = '/home/yunjae_heo/SSD/yunjae.heo/tooth_crop'
save_path = '/home/yunjae_heo/SSD/yunjae.heo/tooth_crop'

f = open(os.path.join(dir_path, 'train.txt'), 'w')

train_anomaly = glob(os.path.join(dir_path, 'data', '0/*.jpg'))
# print(train_anomaly)

train_normal = glob(os.path.join(dir_path, 'data', '1/*.jpg'))
# print(train_normal)
random.shuffle(train_normal)

new_train = train_anomaly + train_normal[:20]
print(new_train)

for line in new_train:
    f.write(line+'\n') 
f.close()

f = open(os.path.join(dir_path, 'test.txt'), 'w')

train_anomaly = glob(os.path.join(dir_path, 'test_data', '0/*.jpg'))
# print(train_anomaly)

train_normal = glob(os.path.join(dir_path, 'test_data', '1/*.jpg'))
# print(train_normal)
random.shuffle(train_normal)

new_train = train_anomaly + train_normal[:10]
print(new_train)

for line in new_train:
    f.write(line+'\n') 
f.close()