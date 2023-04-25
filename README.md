## Getting Started

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
