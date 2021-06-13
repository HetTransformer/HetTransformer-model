# This folder contains code for collecting the dataset and pre-processing them
1. fetch_fakenewsnet_img.py is used to collect the images attached in news content in PolitiFact and GossipCop datasets.
2. visual_feature_extractor.py uses pre-trained ResNet 18 to extract the viusal features from the images attached in news content. This function is used for PolitiFact and GossipCop. We feed in a white image if there's no image attached in news of PolitiFact and GossipCop datasets. All the news in PHEME dataset does not have images, so this function does not apply to PHEME. 
