# PIA

This is the implementation for our group property inference attack against GNNs

## Datasets

The datasets we used in the paper can be download here:

Pokec: https://snap.stanford.edu/data/soc-pokec.html

Facebook: https://snap.stanford.edu/data/ego-Facebook.html

Pubmed: https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes


## GNNs (target model)

The original implemenations of GNN models we used in the paper can be found here:

GCN: https://github.com/tkipf/pygcn

the implementation of both GraphSAGE and GAT from DGL package: https://github.com/dmlc/dgl

## Requirements

- To run the code of GNNs, please use the environments and required packages from the links above:

 - for GCN, use PyTorch 0.4 or 0.5, Python 2.7 or 3.6

 - for GraphSAGE and GAT, import the package of DGL

## Step1: 

run three GNNs on three datasets to get the embeddings, posteriors

run gcn-train.py, gs-train.py, gat-train.py

## Step2: 

run the attack models

for Attack1/2/5/6, run PIA-attak1-attack2.py, PIA-attak5-attack6.py with the embeddings/posteriors from step1

for Attack3/4, run dimension-reduction-tsne.py, dimension-reduction-pca.py, dimension-reduction-encoder.py
 
## Step3: 

evaluate the defense mechanisms

for Noisy embedding/posterior, run defense-laplace.py

for Embedding truncation, run defense-embedding-truncation.py

for OTHER methods we try, PCA dimension reduction: run defense-pca.py, embedding normalizartion: run defense-normalization-softmax.py

## Additional results

We also have some other results, such as the TSNE visualization of the distribution of node embeddings and target model outputs by GCN model on Pokec dataset, these results are included in Results.pdf



