# ImDrug: A Benchmark for Deep Imbalanced Learning in AI-aided Drug Discovery

ImDrug is an open-source and systematic benchmark targeting deep imbalanced learning in AI-aided Drug Discovery. ImDrug features modularized components including formulation of learning setting and tasks, dataset curation, standardized evaluation, and baseline algorithms.

## Installation

### Using `conda`

```bash
conda env create -f environment.yml
conda activate ImDrug
pip install git+https://github.com/bp-kelley/descriptastorus
```
## Configuration
A task can be completely specified with an individual JSON file shown below. It updates the default configuration in `/lib/config/default.py`. Sample JSON files for producing the results in the paper can be found in `/configs/`.
```javascript
{
    "dataset": {
        "drug_encoding": "DGL_GCN", 
        "protein_encoding": "Transformer", 
        "tier1_task": "multi_pred", 
        "tier2_task": "DTI", 
        "dataset_name": "SBAP",
        "split":{
            "method": "random",
            "by_class": true
        }
    },
    "baseline": "Remix_cls",
    "test": {
        "exp_id": "sbap_DGL_GCN_CrossEntropy_0_MLP_2022-06-09-00-10-53-662280"
    },
    "setting": {
        "type": "LT Classification", 
    },
    "use_gpu": true,
    "save_step": 5,
    "show_step": 5,
    "valid_step": 1
}
```

### Data Processing
The 'dataset' entry in the JSON file specifies the dataset to be used, as well as the correponding data processing method with specifications such as featurization and data split. The configuration can be choosen as follows:

- <b>'drug_encoding'</b> determines how the drugs/small-molecule compounds (if any) in the dataset will be featurized.  
``'drug_encoding'``: ["Morgan", "Pubchem’, "Daylight", "rdkit_2d_normalized", "ESPF", "CNN", "CNN_RNN", "Transformer", "MPNN", "ErG", "DGL_GCN", "DGL_NeuralFP", "DGL_AttentiveFP" "DGL_GIN_AttrMasking", "DGL_GIN_ContextPred"]
- <b>'protein_encoding'</b> determines how the proteins/large-molecules (if any) in the dataset will be featurized.  
``'protein_encoding'``: ["AAC", "PseudoAAC", "Conjoint_triad", "Quasi-seq", "ESPF", "CNN", "CNN_RNN", "Transformer"]
- <b>'tier1_task' </b>specifies the type of prediction problems.  
``'tier1_task'``: ["single_pred", "multi_pred"], both are applicable for hybrid prediction.
- <b>'tier2_task'</b> specifies the type of dataset and the prediction label.    
``'tier2_task'``: ["ADME", "TOX", "QM", "BioAct", "Yields", "DTI", "DDI", "Catalyst", "ReactType"]
- <b>'dataset_name'</b> specifies the dataset name.  
``'dataset_name'``: ["BBB_Martins", "Tox21", "HIV", "QM9", "USPTO-50K", "USPTO-Catalyst", "USPTO-1K-TPL", "USPTO-500-MT", "USPTO-Yields", "SBAP", "DrugBank"]
    - WARNING: Note that we keep the original format of "USPTO-500-MT" from [Lu et al.](https://yzhang.hpc.nyu.edu/T5Chem/), for which we have confirmed with the authors that class 334 is missing. To use the dataset properly, one would need to make the class labels consecutive. 
    - WARNING: note that in principle, the yield of "USPTO-Yields" ranges from 0-1. However, the original copy of "USPTO-Yields" from [TDC](https://tdcommons.ai/) contains samples with negative yields or yields above 1, which we exclude in the current version.

- <b>'split.method'</b> specifies the way to split the data, some of which rely on specific domain annotations such as scaffold and time splits.  
``'split.method'``: ["standard", "random", "scaffold", "time", "combination", "group", "open-random", "open-scaffold", "open-time", "open-combination", "open-group"], methods starting with "open-" are reserved for Open LT setting only.

### Imbalanced Learning Algorithms
The configuration of algorithms for imbalanced learning can be choosen as follows:  
- For ``LT Classification`` and ``Imbalanced Classification``:    
``'baseline'``: ["Default_cls", "BalancedSoftmax", "ClassBalanced", "CostSensitive", "InfluenceBalanced", "Mixup_cls", "Remix", "BBN_cls", "CDT", "Decoupling", "DiVE"]  
- For ``Imbalanced Regression``:  
``'baseline'``: ["Default_reg", "Mixup_reg", "Remix_reg", "BBN_reg", "Focal-R", "FDS", "LDS"]  
- For ``Open LT``:  
``'baseline'``: ["Default_cls", "BalancedSoftmax", "ClassBalanced", "InfluenceBalanced", "Remix", "BBN_cls", "OLTR", "IEM"]  
Note that the suffix "cls" and "reg" indicate that the algorithm can be applied for both classification and regression tasks, respectively.

## Run in Docker

To run in Docker, go to `./script/docker`. First download [Miniconda3-latest-Linux-x86_64.sh](https://repo.anaconda.com/miniconda/) and save it to `./common`. Then run `docker build . -t imdrug` within that directory to build the Docker image tagged with the name imdrug. As an example, you can then run the container interactively with a bash shell with `docker run --rm --runtime=nvidia -it -v [PATH_TO_ImDrug]:/root/code imdrug:latest /bin/bash`.

## Running Examples

Note that for the following examples, before running ``python3 script/test.py`` for inference, make sure to update cfg["test"]["exp_id"] in the JSON file to specify the experiment id and the saved model to be tested. 

### LT Classifcation on single_pred.HIV (num_class = 2):

#### Baseline (CrossEntropy)

```bash
python3 script/train.py --config ./configs/single_pred/LT_Classification/baseline/HIV.json
python3 script/test.py --config ./configs/single_pred/LT_Classification/baseline/HIV.json
```

#### Remix

```bash
python3 script/train.py --config ./configs/single_pred/LT_Classification/information_augmentation/Remix/HIV.json
python3 script/test.py --config ./configs/single_pred/LT_Classification/information_augmentation/Remix/HIV.json
```

### LT Classifcation on multi_pred.SBAP (num_class = 2):

#### Baseline (CrossEntropy)

```bash
python3 script/train.py --config ./configs/single_pred/LT_Classification/baseline/SBAP.json
python3 script/test.py --config ./configs/single_pred/LT_Classification/baseline/SBAP.json
```

#### BBN

```bash
python3 script/train.py --config ./configs/single_pred/LT_Classification/module_improvement/BBN/SBAP.json
python3 script/test.py --config ./configs/single_pred/LT_Classification/module_improvement/BBN/SBAP.json
```

### LT Classification on single_pred.UPSTO-50k (num_class = 10):

#### Baseline (CrossEntropy)

```bash
python3 script/train.py --config ./configs/single_pred/LT_Classification/baseline/USPTO-50k.json
python3 script/test.py --config ./configs/single_pred/LT_Classification/baseline/USPTO-50k.json
```

#### BalancedSoftmaxCE

```bash
python3 script/train.py --config ./configs/single_pred/LT_Classification/class-re-balancing/BalancedSoftmaxCE/USPTO-50k.json
python3 script/test.py --config ./configs/single_pred/LT_Classification/class-re-balancing/BalancedSoftmaxCE/USPTO-50k.json
```

### LT Classification on multi_pred.UPSTO-50k (num_class = 10):

#### Baseline (CrossEntropy)

```bash
python3 script/train.py --config ./configs/multi_pred/LT_Classification/baseline/USPTO-50k.json
python3 script/test.py --config ./configs/multi_pred/LT_Classification/baseline/USPTO-50k.json
```

#### Decoupling

```bash
python3 script/train.py --config ./configs/multi_pred/LT_Classification/module_improvement/Decoupling/USPTO-50k.json
python3 script/test.py --config ./configs/multi_pred/LT_Classification/module_improvement/Decoupling/USPTO-50k.json
```

### LT Regression on single_pred.QM9

#### Baseline (MSE)

```bash
python3 script/train.py --config ./configs/single_pred/LT_Regression/baseline/QM9.json
python3 script/test.py --config ./configs/single_pred/LT_Regression/baseline/QM9.json
```

#### LDS

```bash
python3 script/train.py --config ./configs/single_pred/LT_Regression/LDS/QM9.json
python3 script/test.py --config ./configs/single_pred/LT_Regression/LDS/QM9.json
```

### LT Regression on multi_pred.SBAP

#### Baseline (MSE)

```bash
python3 script/train.py --config ./configs/multi_pred/LT_Regression/baseline/SBAP.json
python3 script/test.py --config ./configs/multi_pred/LT_Regression/baseline/SBAP.json
```

#### FDS

```bash
python3 script/train.py --config ./configs/multi_pred/LT_Regression/FDS/SBAP.json
python3 script/test.py --config ./configs/multi_pred/LT_Regression/FDS/SBAP.json
```

### Open LT on multi_pred.Drugbank

#### Baseline (CrossEntropy)

```bash
python3 script/train.py --config ./configs/multi_pred/Open_LT/baseline/Drugbank.json
python3 script/test.py --config ./configs/multi_pred/Open_LT/baseline/Drugbank.json
```

#### OLTR

```bash
python3 script/train.py --config ./configs/multi_pred/Open_LT/OLTR/Drugbank.json
python3 script/test.py --config ./configs/multi_pred/Open_LT/OLTR/Drugbank.json
```

### Training output
Each training process will generate a log (e.g., BBB_Martins_DGL_GCN_Transformer_MLP_2022-04-28-20-30.log) in `./output/${DATASET_NAME}/logs`, and the models in `./output/${DATASET_NAME}/models/${EXP_ID}`. 

### Testing output

Note that before testing, you need to specify the training experiment id in cfg['test']['exp_id']. Each testing process will generate a log and a .pdf image of confusion matrix (e.g., BBB_Martins_Transformer_Transformer_MLP_2022-05-09-11-55.pdf) in `./output/${DATASET_NAME}/test`.

### Testing trained models of a dataset all at once

To test trained models all at once, specify the "root_path" in `./test_all.py` by the directory where all training logs are stored, i.e., ``root_path = ./output/${DATASET_NAME}/logs``. Then run the following command line

```bash
python3 test_all.py 
```

## Benchmarks

### LT Classification 

![](https://storage.googleapis.com/imdrug_data/Figures/table2_latest.png)

### LT Regression

![](https://storage.googleapis.com/imdrug_data/Figures/table4_latest.png)

### Open LT

![](https://storage.googleapis.com/imdrug_data/Figures/table3_latest.png)

### Results on Class Subsets

![](https://storage.googleapis.com/imdrug_data/Figures/figure2.png)

### Results on Out-of-distribution (OOD) Splits

![](https://storage.googleapis.com/imdrug_data/Figures/figure4.png)

<!-- ### LT Classifcation (Metrics: balanced accuracy (BA), balanced-f1 (BF))
**All results are on test set unless specified otherwise.**
#### BBB_Martins (num = 2)
| Task        | Algo Type                | Backbone | Split | Train BA (200 epochs) | BA | BA-head | BA-mid | BA-tail | BA-open | BF | BF-head | BF-mid | BF-tail | BF-open |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_pred | baseline (lr=0.1)     | DGL_GCN       | random | 0.7458 | 0.6984 | 0.8968 | N/A | 0.5 | N/A | 0.686 | 0.7483 | N/A | 0.6237 | N/A |
| single_pred | baseline (lr=0.1)     | Transformer   | random | 0.7835 | 0.6833 | 0.9290 | N/A | 0.4375 | N/A | 0.6629 | 0.7458 | N/A | 0.5801 | N/A |                             
#### USPTO-50k (num = 10, lt_frac=[0.4, 0.4, 0.2])
| Task          | Algo Type                | Backbone | Split | Train BA (200 epochs) | BA | BA-head | BA-mid | BA-tail | BA-open | BF | BF-head | BF-mid | BF-tail | BF-open |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_pred | baseline (lr=0.1)      | DGL_GCN  | random | 0.3384 | 0.3230 | 0.5003 | 0.2296 | 0.1551 | N/A | 0.3166 | 0.3582 | 0.3004 | 0.2660 | N/A |                          
| single_pred | MI.PBS/MI.DRS (lr=0.1) | DGL_GCN  | random | 0.4005 | 0.3833 | 0.473- | 0.3397 | 0.2910 | N/A | 0.3913 | 0.3823 | 0.3814 | 0.4288 | N/A |                         
| multi_pred  | Baseline (lr=0.1)      | DGL_GCN  | random | 0.4876 | 0.4338 | 0.5745 | 0.4218 | 0.1765 | N/A | 0.4366 | 0.4278 | 0.5168 | 0.2939 | N/A |                         
| multi_pred  | MI.PBS/MI.DRS (lr=0.1) | DGL_GCN  | random | 0.5256 | 0.4770 | 0.5605 | 0.4848 | 0.2944 | 
### LT Regression
| Dataset                 | Type                | Method                           |  Balanced MSE (200 epochs) |
| --- | --- | --- | --- |
| multi_pred.BindingDB_Kd | Baseline            | MSE                              |  NAN                       |
| multi_pred.BindingDB_Kd | Class Re-balancing  | Balanced MSE                     |                            | -->

## Datasets

ImDrug is hosted on Google Cloud, each of the data can be accessed via https://storage.googleapis.com/imdrug_data/{$DATASET_NAME}.

Complete list of dataset_names:
- bbb_martins.tab
- hiv.tab
- tox21.tab
- qm9.csv
- sbap.csv
- drugbank.csv
- uspto_1k_TPL.csv
- uspto_500_MT.csv
- uspto_50k.csv
- uspto_catalyst.csv
- uspto_yields.csv

## Cite Us
Coming soon.

## License
ImDrug codebase is under the MIT license. For individual dataset usage, the dataset license will come up soon.

## Contact

Reach us at imdrugbenchmark@gmail.com or open a GitHub issue.
