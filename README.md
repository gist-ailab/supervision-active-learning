## Getting Started

### Overview

Image Classification with additional Supervision for selected data

선별된 데이터에 대한 추가 지도를 통한 이미지 분류

현재 추가 지도의 대상으로 Bbox 정보를 사용하여, ImageNet의 Annotation과 같은 형식을 가진 데이터들에 대해서 적용 가능

box_loss : 추가 지도를 위해 Bbox의 위치 크기 정보를 활용

point_loss : 추가 지도를 위해 Bbox의 위치 정보를 활용

### Repository Structure
```
supervision-active-learning
├── box_loss
       └── dataset.py
       └── img_save.py
       └── main_random10.py
       └── main_selected10.py
       └── resnet.py
       └── utils.py
       └── etc(deprecated)
├── point_loss
       └── dataset.py
       └── img_save.py
       └── main_random10.py
       └── main_selected10.py
       └── resnet.py
       └── utils.py
└── LOC_synset_mapping 
└── new_LOC_train_solution_30 
└── new_LOC_test_solution_30 
└── new_LOC_val_solution_30 
└── etc(for data visualization)
```

### Environment Setup

Install dependencies
```
sudo apt update && sudo apt upgrade
conda create -n env_name
conda activate env_name
conda install -c anaconda git
git clone https://github.com/gist-ailab/supervision-active-learning.git
conda env update --file requirements.yaml --prune
```

## Train & Evaluation

### Dataset Preparation
1. Download `ILSVRC' from MAT or Connect to MAT.
```
PATH : ailab_mat/dataset/ILSVRC
```
2. Organize the folders as follows
```
ILSVRC
├── ILSVRC (Original folder)
       └── Annotations
       └── Data
       └── ImageSets
└── LOC_synset_mapping 
└── new_LOC_train_solution_30 
└── new_LOC_test_solution_30 
└── new_LOC_val_solution_30 
└── ... (useless)
```
### Train and Test on sample dataset
```
python supervision_active_learning/box_loss/main_selected10.py --seed 99 --gpu 4 --data_path DATA_PATH --save_path SAVE_PATH --epoch 50 --epoch2 50 --loss loss4 --batch_size 32 --lr 0.01
```
