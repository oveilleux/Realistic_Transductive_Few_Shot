# Realistic evaluation of transductive few-shot learning


##  Introduction
This repo contains the code for our NeurIPS 2021 submitted paper "Realistic evaluation of transductive few-shot learning". This is a framework that regroups all methods evaluated in our paper except for SIB and LR-ICI. Results provided in the paper can be reproduced with this repo. Code was developed under python 3.8.3 and pytorch 1.4.0.


## 1. Getting started


### 1.1 Quick installation (recommended) (Download datasets and models)
To download datasets and pre-trained models (checkpoints), follow instructions 1.1.1 to 1.1.2 of NeurIPS 2020 paper "TIM: Transductive Information Maximization" public implementation (https://github.com/mboudiaf/TIM)

#### 1.1.1 Place datasets
Make sure to place the downloaded datasets (data/ folder) at the root of the directory.

#### 1.1.2 Place models
Make sure to place the downloaded pre-trained models (checkpoints/ folder) at the root of the directory.

### 1.2 Manual installation
Follow instruction 1.2 of NeurIPS 2020 paper "TIM: Transductive Information Maximization" public implementation (https://github.com/mboudiaf/TIM) if facing issues with previous steps. Make sure to place data/ and checkpoints/ folders at the root of the directory.

### 2. Requirements
To install requirements:
```bash
conda create --name <env> --file requirements.txt
```
Where \<env> is the name of your environment

## 3. Reproducing the main results

Before anything, activate the environment:
```python
source activate <env>
```

### 3.1 Table 1 and 2 results in paper

To reproduce the results from Table 1. and 2. in the paper, from the root of the directory execute this python command.
```python
python3 -m src.main --base_config <path_to_base_config_file> --method_config <path_to_method_config_file> 
```

The <path_to_base_config_file> follows this hierarchy:
```python
config/<balanced or dirichlet>/base_config/<resnet18 or wideres>/<mini or tiered or cub>/base_config.yaml
```

The <path_to_method_config_file> follows this hierarchy:
```python
config/<balanced or dirichlet>/methods_config/<alpha_tim or baseline or baseline_pp or bdcspn or entropy_min or laplacianshot or protonet or pt_map or simpleshot or tim>.yaml
```

For instance, if you want to reproduce the results in the **balanced** setting on mini-Imagenet, using ResNet-18, with alpha-TIM method go to the root of the directory and execute:
```python
python3 -m src.main --base_config config/balanced/base_config/resnet18/mini/base_config.yaml --method_config config/balanced/methods_config/alpha_tim.yaml
```

If you want to reproduce the results in the **randomly balanced** setting on mini-Imagenet, using ResNet-18, with alpha-TIM method go to the root of the directory and execute:
```python
python3 -m src.main --base_config config/dirichlet/base_config/resnet18/mini/base_config.yaml --method_config config/dirichlet/methods_config/alpha_tim.yaml
```

# Reusable data sampler module
One of our main contribution is our realistic task sampling method following Dirichlet's distribution. 
![plot](plots/dirichlet_density_plot.png)

Our realistic sampler can be found in [sampler.py](src/datasets/sampler.py) file. The sampler has been implemented following Pytorch's norms and in a way that it can be easily reused and integrated in other projects.

The following notebook [exemple_realistic_sampler.ipynb](exemple_realistic_sampler.ipynb) is an exemple that shows how to initialize and use our realistic category sampler.

# Contact
For further questions or details, reach out to Olivier Veilleux (olivier.veilleux.2@ens.etsmtl.ca)

# Acknowledgements
Special thanks to the authors of NeurIPS 2020 paper "TIM: Transductive Information Maximization" (TIM) (https://github.com/mboudiaf/TIM) for publicly sharing their pre-trained models and their source code from which this repo was inspired from.


