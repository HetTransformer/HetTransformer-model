# HetTransformer-model
This repository introduces the use of HetTransformer model.
![avatar](/figures_and_tables/figure.png)

## Directory Structure
This is the intended directory sturcture after the completion of data collection and processing steps.
```
.
├── data/                               # Datasets and their processing scripts
|   ├── FakeNewsNet/                    # Cloned FakeNewsNet repository
|   ├── PHEME/                          # Collected and unzipped PHEME datapath
|   ├── processed_data/                 # Pre-processed data
|   |   ├── FakeNewsNet/                # Pre-processed FakeNewsNet data
|   |   └── PHEME/                      # Pre-processed PHEME data
|   ├── rwr_results/                    # Generated RWR neighbors
|   ├── README.md                       # Data pre-processing instructions
|   └── ...                             # Data pre-processing scripts
├── figures_and_tables/                 # Figures and tables in this README.md 
├── models/                             # Experiments-related scripts
|   ├── train_and_evaluation/           # The model training and evaluation code
|   ├── para_sensitivity/               # Parameter sensitivity code
|   ├── data_splits/                    # Train-val-test split used
|   ├── best_models/                    # The reserved model for users to run the training script to save their models
|   └── pre-trained/                    # The pre-trained models
├── README.md                           # Reproduction instructions
└── requirements.txt                    # Dependencies
```
Run the following commands to creat the directory sturcture needed.
```
mkdir data/processed_data data/rwr_results
```

## 0. Requirements
This repository is coded in `python==3.8.5`.
Please run the following command to install the other requirements from `requirements.txt`.
```
pip install -r requirements.txt
```

## 1. Dataset Collection
Three datasets, PolitiFact, GossipCop and PHEME are used. While the collection of the first two takes many days, the last one can be done in minutes.

### Collect PolitiFact and GossipCop
To compile with [Twitter Developer Policy](https://developer.twitter.com/en/developer-terms/policy), Twitter datasets cannot be shared. Hence, each developer must crawl their own copies of FakeNewsNet for PoliFact and GossipCop datasets. 

First of all, run the following to get a copy of FakeNewsNet under the `data/` directory.
```
cd data
git clone https://github.com/KaiDMML/FakeNewsNet
cd ..
```
Then, please follow the steps in [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet).
!! Due to the Twitter API key limits, it may take more than 20 days to collect a complete set of FakeNewsNet if you only have one Twitter API key. To verify the collection, you may follow the instructions in [`data/README.md`](https://github.com/HetTransformer/HetTransformer-model/tree/main/data).

### Collect PHEME
Run the following on command line to collect PHEME under `data/`, unzip it, and rename it.
```
cd data
wget -O PHEME.tar.bz2 "https://ndownloader.figshare.com/files/6453753"
tar -vxf PHEME.tar.bz2
mv pheme-rnr-dataset PHEME
cd ..
```
The zipped file is only 25M and can be downloaded in around 3 minutes.

## 2. Data Pre-processing
Data pre-processing includes image fetching, image encoding, text encoding, graph construction, and the extraction of other features.
The details are described in [`data/README.md`](https://github.com/HetTransformer/HetTransformer-model/tree/main/data).

## 3. Model Training and evaluation
After generating batch files following step 1 and 2 in `data/processed_data/FakeNewsNet/PolitiFact/batch/`; `data/processed_data/FakeNewsNet/GossipCop/batch/`; `data/processed_data/PHEME/batch/` respectively. 

Run scripts in `models/train_and_evaluation/` to train the model and get the evaluation results. 

You can also load the pre-trained models in `models/pre-trained/` following the evaluation scripts decribed under `models/train_and_evaluation/`.

The data splits we used is presented in ` models/data_splits/`. The best models generated will be saved under `models/best_models/`

## 4. Results
![avatar](/figures_and_tables/table.png)
