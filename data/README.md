# This folder contains code for collecting the dataset and pre-processing them
## Data collection
### PolitiFact and FakeNewsNet
Due to Twitter Developer Policy, Twitter datasets cannot be shared. Hence, each developer must crawl their own copies of datasets.
```
git clone https://github.com/KaiDMML/FakeNewsNet.git
```

### PHEME
```
wget https://ndownloader.figshare.com/files/6453753
```

## Image fetching and encoding
* `fetch_fakenewsnet_img.py` is used to collect the images attached in news content in PolitiFact and GossipCop datasets.
* `fetch_fakenewsnet_img_multiproc.py` is used to collect the images attached in news content in PolitiFact and GossipCop datasets with multiple workers.
* `visual_feature_extractor.py` uses pre-trained ResNet 18 to extract the viusal features from the images attached in news content. This function is used for PolitiFact and GossipCop. We feed in a white image if there's no image attached in news of PolitiFact and GossipCop datasets. All the news in PHEME dataset does not have images, so this function does not apply to PHEME. 

## Graph constuction and random walk with restarts
* `generate_graph_fakenewsnet.py` generates the graph input needed from PolitiFact and GossipCop datasets for `random_walk_with_restart.py` with multi-processing.
* `generate_graph_pheme.py` generates the graph input needed from PHEME dataset for `random_walk_with_restart.py` with multi-processing.
* `random_walk_with_restart.py` performs _random walk with restarts_ (RWR) for PolitiFact, GossipCop and PHEME with multi-processing.

## Text encoding
* `text_embedder.py` defines a Transformer-based text embedder for all text embedding.
* `embed_text_fakenewsnet.py` embeds the text for all types of nodes in PolitiFact and GossipCop datasets with `text_embedder.py` with multi-processing.
* `embed_text_pheme.py` embeds the text for all types of nodes in PHEME datasets with `text_embedder.py` with multi-processing.
