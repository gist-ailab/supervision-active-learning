# README TEMPLATES
- @@

## Updates & TODO Lists
- @@

## Getting Started

### Environment Setup

1. Install dependencies
```
sudo apt update && sudo apt upgrade
```

2. Set up a python environment
```
conda create -n test_env python=3.8
conda activate test_env
pip install -r requirement.txt
```
## Train & Evaluation

### Dataset Preparation
1. Download `sample dataset' from MAT.
```
wget sample_dataset.com
```
2. Extract it to `sample folder`
```
tar -xvf sample_dataset.tar
```
3. Organize the folders as follows
```
test
├── output
└── datasets
       └── sample_dataset
              └──annotations
              └──train
              └──val       
```
### Train on sample dataset
```
python train_net.py --epoch 100
```

### Evaluation on test dataset
```
python test_net.py --vis_results
```

## License

The source code of this repository is released only for academic use. See the [license](./LICENSE.md) file for details.

## License
Distributed under the MIT License.

## Acknowledgments
This work was supported by Institute for Information & Communications Technology Promotion(IITP) grant funded by Korea goverment(MSIT) (No.2019-0-01335, Development of AI technology to generate and validate the task plan for assembling furniture in the real and virtual environment by understanding the unstructured multi-modal information from the assembly manual.
