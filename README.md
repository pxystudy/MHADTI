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
Researchers could employ MHADTI model on their own datasets to predict DTIs, there are three steps for running MHADTI model listed below. The whole workflow can be seen in Figure 1 in the manuscript.
* Firstly, multisource data related with drugs and targets needs to be collected. The data mainly contains MACCS molecular fingerprint of drugs, drug side effects, drug-target interaction data, protein sequence, Gene Ontology annotation data of proteins, and domains of proteins.
* Secondly, multiview heterogeneous information networks need to be constructed. The similarity of between drugs (or targets) can be evaluated by various models (see our manuscript). Further, different drugs and target similarity networks can be established. Finally, multiview heterogeneous information networks can be formed with the drug similarity networks, target similarity networks and the drug-target interaction network. 
* Thirdly, all the multiview heterogeneous information networks can be fed into MHADTI model for training. After that, MHADTI could predict novel drug-target interactions.

# Thanks!
