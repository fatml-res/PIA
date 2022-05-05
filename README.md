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

To run the code of GNNs, please use the environments and required packages from the links above

## Step1: 

run three GNNs on three datasets to get the embeddings, posteriors

run gcn-pokec-train.py, gcn-fb-train.py, gcn-pubmed-train.py, gs-pokec-train.py, gs-fb-train.py, gs-pubmed-train.py, gat-pokec-train.py, gat-fb-train.py, gat-pubmed-train.py

## Step2: 

run Attack1/2/3/4/5/6 with the embeddings/posteriors

run 

## Step3: 

evaluate the defense mechanisms


## Additional results

We also have some other results, such as the TSNE visualization of the distribution of node embeddings and target model outputs by GCN model on Pokec dataset, these results are included in Results.pdf



