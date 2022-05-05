import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
import pickle as pkl
import networkx as nx
import numpy as np
import random


def process_pokec(data_dir,ii,sed):
    # data_dir='./'
    f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    feat_data = list(ft[:,1:])

    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index],dtype='int64')


    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def __getitem__(self, i):
    return self.graph

def __len__(self):
    return 1



def process_pubmed(data_dir,ii,sed):
    # data_dir='./'
    f2 = open("{0}/pubmed-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    feat_data = list(ft[:, idx])

    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int64')
    print((labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 4000
    n_val = 1000
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_fb(data_dir,ii,sed):
    # data_dir='./'
    f2 = open("{0}/fb-adj-feat-{1}-{2}-edu-gender.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    num_nodes = 1600
    num_feats = 1283

    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, 53)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels = np.array((ft)[:, e_index]-1,dtype='int64')
    print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_pokec_interintra(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pokec-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    feat_data = list(ft[:,2:])
    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    gender_index = 1
    pub_index = 0
    age_index = 2
    height_index = 3
    weight_index = 4
    region_index = 5

    labels = np.array((ft)[:, pub_index],dtype='int64')

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)

def process_pubmed_interintra(data_dir,ii,sed):
    # data_dir='./'
    # f2 = open("{0}/pokec-adj-feat-{1}-{2}-gender-original.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    f2 = open("{0}/pubmed-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    adj1 = np.array(adj.todense())

    # g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = np.shape(ft)[1]
    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    idx = np.delete(idx, num_feats - 1)

    feat_data = list(ft[:, idx])

    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)

    labels = np.array((ft)[:, num_feats - 1] - 1, dtype='int64')
    print((labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = int(n_nodes * 0.7)
    n_val = int(n_nodes * 0.1)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train :] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)


def process_fb_interintra(data_dir,ii,sed):
    # data_dir='./'
    f2 = open("{0}/fb-adj-feat-{1}-{2}.pkl".format(data_dir, str(ii), str(sed)), 'rb')
    adj, ft = pkl.load(f2, encoding='latin1')

    g = nx.Graph(adj)

    num_nodes = np.shape(ft)[0]
    num_feats = 1283

    # adj1 = np.array(adj.todense())

    idx = np.arange(num_feats)
    # idx = np.delete(idx, 77)
    idx = np.delete(idx, 53)

    feat_data = list(ft[:, idx])
    print(np.shape(feat_data))

    # print((ft))
    src=[]
    dst=[]
    wt=[]
    for eg in g.edges():
        src.append(eg[0])
        dst.append(eg[1])
        wt.append(1)



    g_index = 77  # gender
    r_index = 1154  # religion
    p_index = 1278  # political
    e_index = 53  # education_type

    labels = np.array((ft)[:, e_index]-1,dtype='int64')
    print(np.shape(labels))

    # nodes_data = pd.read_csv('./members.csv')
    # edges_data = pd.read_csv('./interactions.csv')
    node_features = torch.from_numpy(np.array(feat_data,dtype='float32'))
    node_labels = torch.from_numpy(labels)
    edge_features = torch.from_numpy(np.array(wt,dtype='float32'))
    edges_src = torch.from_numpy(np.array(src,dtype='int32'))
    edges_dst = torch.from_numpy(np.array(dst,dtype='int32'))

    graph = dgl.graph((edges_src, edges_dst), num_nodes=ft.shape[0])
    graph.ndata['feat'] = node_features
    graph.ndata['label'] = node_labels
    graph.edata['weight'] = edge_features

    # If your dataset is a node classification dataset, you will need to assign
    # masks indicating whether a node belongs to training, validation, and test set.
    n_nodes = ft.shape[0]
    n_train = 1200
    n_val = 100
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask2 = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    test_mask2[n_train:] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['test_mask2'] = test_mask2

    return (graph)



