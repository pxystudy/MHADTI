# MHADTI
MHADTI: The codes demo for paper "MHADTI:Predicting drug-target interactions via multiview heterogeneous information network embedding with hierarchical attention mechanisms".

# Required Packages
* Python 3.9.7
* Pytorch 1.10.0
* Numpy 1.21.4
* Pytorch-geometric 2.0.2
* Pandas 1.3.5
* Scikit-learn 1.0.1
* Scipy 1.7.3

# About Data
* heterogeneous: three heterogeneous graph obtained by constructing and fusing similarity networks
* heterogeneous/DTD.txt: the adjacency matrix based on meta-path drug-target-drug
* drug.txt: list of drug names
* target.txt: list of target names
* drug_feature.txt: initial feature of the corresponding drug name
* target_feature.txt: initial feature of the corresponding target name
* DT_interaction.txt: all the drug-target interactions
* Anyone can download this datasets and preprocess it as described in the paper. If there is any more question about dataset, please feel free to contact one of authors Peng Xiangyu.

# Quick start
* First, you need to unzip the .zip folder in data to get the datasets for training.
* Then, run the code "main.py" directly.

# Data Preprocessing
If you want use MHADTI model in other datasets, you need to do the following processing on the obtained DTI datasets.
* First, we need to find drug-related data: MACCS nolecular fingerprint of drugs and drug side effects; Protein-related data: protein sequence, annotated terms of protein on the three gene ontology substructures, and protein domains.
* Secondly, calculate the node similarity according to the drug and target similarity calculation methods in the Method part of paper.
* Thirdly, consturct heterogeneous networks according to the model frame digram of the paper.

# Thanks!
