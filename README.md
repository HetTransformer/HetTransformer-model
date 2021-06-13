# HetTransformer-model
This repository introduces the use of HetTransformer model.

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
wget -O pheme-figshare.tar.bz2 "https://ndownloader.figshare.com/files/6453753"
tar -vxf pheme-figshare.tar.bz2
mv pheme-rnr-dataset pheme-figshare
cd ..
```
The zipped file is only 25M and can be downloaded in around 3 minutes.

## 2. Data Pre-processing
Data pre-processing includes image fetching, image encoding, text encoding, graph construction, and the extraction of other features.
The details are described in [`data/README.md`](https://github.com/HetTransformer/HetTransformer-model/tree/main/data).

## 3. Model Training and evaluation
## 4. Results
