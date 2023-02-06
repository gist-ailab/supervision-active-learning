# import csv
# import os

# dir_path = '/home/yunjae_heo/SSD/yunjae.heo/ILSVRC'
# save_path = '/home/yunjae_heo/SSD/yunjae.heo/ILSVRC'
# num_class = 100

# f = open(os.path.join(dir_path, 'LOC_synset_mapping.txt'), 'r')
# mapping = f.readlines()

# train = open(os.path.join(dir_path, 'LOC_train_solution.csv'), 'r')
# train_csv = csv.reader(train)
# train_csv = list(train_csv)
# val = open(os.path.join(dir_path, 'LOC_val_solution.csv'), 'r')
# val_csv = csv.reader(val)
# val_csv = list(val_csv)

# new_train = open(os.path.join(save_path, f'LOC_train_solution_{num_class}.csv'), 'w')
# train_writer = csv.writer(new_train)
# new_val = open(os.path.join(save_path, f'LOC_val_solution_{num_class}.csv'), 'w')
# val_writer = csv.writer(new_val)

# train_writer.writerow(['ImageId', 'PredictionString'])
# val_writer.writerow(['ImageId', 'PredictionString'])

# # next(train_csv)
# # next(val_csv)

# count = 0
# for line in mapping:
#     count += 1
#     label = line.split(' ')[0]
#     print(label)
#     for t_line in train_csv:
#         t_label = t_line[0].split('_')[0]
#         if t_label == label:
#             # print(t_label)
#             train_writer.writerow(t_line)
#     for v_line in val_csv:
#         v_label = v_line[1].split(' ')[0]
#         if v_label == label:
#             val_writer.writerow(v_line)
#     print(count)
#     if count == num_class: break

# f.close()
# train.close()
# val.close()
# new_train.close()
# new_val.close()

import csv
import os
import random


dir_path = '/home/yunjae_heo/SSD/yunjae.heo/ILSVRC'
save_path = '/home/yunjae_heo/SSD/yunjae.heo/ILSVRC'

f = open(os.path.join(dir_path, f'LOC_train_solution_30.csv'), 'r')
train_csv = csv.reader(f)

temp_list = []
for t_line in train_csv:
    temp_list.append(t_line)

random.shuffle(temp_list)

val = temp_list[:1500]
train = temp_list[1500:]

new_train = open(os.path.join(save_path, f'new_LOC_train_solution_30.csv'), 'w')
train_writer = csv.writer(new_train)
new_val = open(os.path.join(save_path, f'new_LOC_val_solution_30.csv'), 'w')
val_writer = csv.writer(new_val)

for data in train:
    train_writer.writerow(data)
for data in val:
    val_writer.writerow(data)
    
f.close()
new_train.close()
new_val.close()