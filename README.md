# PIA

This is the implementation for our group property inference attack against GNNs, which has been submitted to ACM CCS 2022

## Datasets

The datasets we used in the paper can be download here:

- Pokec: https://snap.stanford.edu/data/soc-pokec.html

- Facebook: https://snap.stanford.edu/data/ego-Facebook.html

- Pubmed: https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes

The sampled sub-graphs for PIA can be downloaded in this link: 

- For each property task, we have 1000 sub-graphs to do training and testing, and the training/testing split ratio is 0.7/0.3.


## GNNs (target model)

The original implemenations of GNN models we used in the paper can be found here:

- GCN: https://github.com/tkipf/pygcn

- the implementation of both GraphSAGE and GAT from DGL package: https://github.com/dmlc/dgl

## Requirements

To run the code of GNNs, please use the environments and required packages from the links above:

 - for GCN, use PyTorch 0.4 or 0.5, Python 2.7 or 3.6

 - for GraphSAGE and GAT, import the package of DGL

## Step1: 

Run three GNNs on three datasets to get the embeddings, posteriors

run gcn-train.py, gs-train.py, gat-train.py

## Step2: 

Run the attack models

- for Attack1/2/5/6, python PIA-attak1-attack2.py, PIA-attak5-attack6.py with the embeddings/posteriors from step1

- for Attack3/4, run dimension-reduction-tsne.py, dimension-reduction-pca.py, dimension-reduction-encoder.py
 
## Step3: 

Evaluate the defense mechanisms

- For Noisy embedding/posterior, run defense-laplace.py

- For Embedding truncation, run defense-embedding-truncation.py

- For OTHER methods we try, PCA dimension reduction: run defense-pca.py, embedding normalization: run defense-normalization-softmax.py

## Additional results

We also have some other results in Results.pdf which are not included in the paper because of the space limitation, it includes the following contents:

- the TSNE visualization of the distribution of node embeddings and target model outputs by GNN models 

- the defense results of embedding normalization

- the results of Influence scores of different node/link groups on three dataset (additional results for Table 6 in the paper)



