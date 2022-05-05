from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data_pokec, accuracy
from models import GCN_pia

from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import pandas as pd
import pickle as pk

# from keras.layers import Input, Dense
# from keras.models import Model

def readembds(file_name):
    file = open(file_name)
    first_line=file.readline()
    first_line = first_line.strip().split(' ')
    first_line = list(map(int, first_line))
    # print(first_line)
    dataMat = []
    for line in file.readlines():
        curLine = line.strip().split('\t')
        floatLine = list(map(float, curLine))
        # print(floatLine)
        dataMat.append(floatLine[0:first_line[1]])
    embeddings = np.array(dataMat)
    embeddings1=np.mean(embeddings,axis=0)
    embeddings2 = np.mean(embeddings, axis=1)
    embeddings3 = np.amax(embeddings, axis=0)
    embeddings4 = np.amax(embeddings, axis=1)
    return (embeddings4)


def readpost(file_name):
    file = open(file_name)
    # first_line=file.readline()
    # first_line = first_line.strip().split(' ')
    # first_line = list(map(int, first_line))
    # print(first_line)
    dataMat = []
    for line in file.readlines():
        curLine = line.strip().split('\t')
        floatLine = list(map(float, curLine))
        # print(floatLine)
        dataMat.append(floatLine)
    embeddings = np.array(dataMat)
    dif = np.zeros(np.shape(embeddings)[0])
    for i in range(np.shape(embeddings)[1]):
        for j in range(i + 1, (np.shape(embeddings)[1])):
            dif += np.absolute(embeddings[:, i] - embeddings[:, j])
    dif = dif / (0.5 * np.shape(embeddings)[1] * (np.shape(embeddings)[1] - 1))
    embeddings5 = dif
    embeddings1 = np.mean(embeddings, axis=0)
    embeddings2 = np.mean(embeddings, axis=1)
    embeddings3 = np.amax(embeddings, axis=0)
    embeddings4 = np.amax(embeddings, axis=1)
    return (embeddings5)


dataset=['pokec','fb','pubmed']
tps=['pokec','fb','pubmed','pokec-interintra','fb-interintra','pubmed-interintra']
feats={}
feats_neg={}
i=0
results_mlp = []
results_lr = []
results_rf = []
tsts_mlp = []
tsts_lr = []
tsts_rf = []
for tp1 in tps:
    label = []
    feat_para1 = []
    feat_para2 = []
    feat_para12 = []
    feat_embed1 = []
    feat_embed2 = []
    feat_embed12 = []
    feat_post = []
    label_neg = []
    feat_para1_neg = []
    feat_para2_neg = []
    feat_para12_neg = []
    feat_embed1_neg = []
    feat_embed2_neg = []
    feat_embed12_neg = []
    feat_post_neg = []

    for sed in range(1, 6):
        for ii in range(-100, 100):
            s = ii + 100
            seed = sed


            file1='./%s/pokec-embed1-%s-%s-12-13-%s.txt' % (tp1,ii, seed,tp1)
            print(file1)
            # embed11,embed12,embed13,embed14=readembds(file1)
            embed1= readembds(file1)


            file2='./%s/pokec-embed2-%s-%s-12-13-%s.txt' % (tp1,ii, seed,tp1)
            # embed21,embed22,embed23,embed24=readembds(file2)
            embed2= readembds(file2)

            file3='./%s/pokec-output_test-%s-%s-12-13-%s.txt' % (tp1,ii, seed,tp1)
            posterior1=readpost(file3)
            if (int(s / 100)) % 2 == 0:
                feat_post_neg.append(np.array(posterior1).flatten())
                feat_embed1_neg.append(np.array(embed1).flatten())
                feat_embed2_neg.append(np.array(embed2).flatten())

            else:
                feat_post.append(np.array(posterior1).flatten())
                feat_embed1.append(np.array(embed1).flatten())
                feat_embed2.append(np.array(embed2).flatten())

    embed1 = np.concatenate((feat_embed1, feat_embed1_neg), axis=0)
    embed2 = np.concatenate((feat_embed2, feat_embed2_neg), axis=0)
    posts = np.concatenate((feat_post, feat_post_neg), axis=0)


    # from sklearn.decomposition import PCA

    from sklearn.decomposition import PCA

    tp = 'pca'

    pca11 = PCA(n_components=744)
    pe11 = pca11.fit_transform(embed1)
    # print(tp0,pca11.explained_variance_ratio_)

    pca21= PCA(n_components=618)
    pe21 = pca21.fit_transform(embed2)
    # print(tp0,pca21.explained_variance_ratio_)

    pca31 = PCA(n_components=756)
    pp1 = pca31.fit_transform(posts)
    # print(tp0,pca31.explained_variance_ratio_)


    pca12 = PCA(n_components=523)
    pe12 = pca12.fit_transform(embed1)
    # print(tp0,pca12.explained_variance_ratio_)

    pca22= PCA(n_components=362)
    pe22 = pca22.fit_transform(embed2)
    # print(tp0,pca22.explained_variance_ratio_)

    pca32 = PCA(n_components=542)
    pp2 = pca32.fit_transform(posts)
    # print(tp0,pca32.explained_variance_ratio_)



    pca13 = PCA(n_components=395)
    pe13 = pca13.fit_transform(embed1)
    # print(tp0,pca13.explained_variance_ratio_)

    pca23= PCA(n_components=235)
    pe23 = pca23.fit_transform(embed2)
    # print(tp0,pca23.explained_variance_ratio_)

    pca33 = PCA(n_components=417)
    pp3 = pca33.fit_transform(posts)
    # print(tp0,pca33.explained_variance_ratio_)

    with open("./{}/embed1-{}-{}-12-13-{}-0.99.pkl".format(tp, str(ii), str(sed),tp1), "wb") as f:
        pk.dump(pe11, f)

    with open("./{}/embed2-{}-{}-12-13-{}-0.99.pkl".format(tp, str(ii), str(sed), tp1), "wb") as f:
        pk.dump(pe21, f)

    with open("./{}/post-{}-{}-12-13-{}-0.99.pkl".format(tp, str(ii), str(sed), tp1), "wb") as f:
        pk.dump(pp1, f)

    with open("./{}/embed1-{}-{}-12-13-{}-0.95.pkl".format(tp, str(ii), str(sed), tp1), "wb") as f:
        pk.dump(pe12, f)

    with open("./{}/embed2-{}-{}-12-13-{}-0.95.pkl".format(tp, str(ii), str(sed), tp1), "wb") as f:
        pk.dump(pe22, f)

    with open("./{}/post-{}-{}-12-13-{}-0.95.pkl".format(tp, str(ii), str(sed), tp1), "wb") as f:
        pk.dump(pp2, f)

    with open("./{}/embed1-{}-{}-12-13-{}-0.9.pkl".format(tp, str(ii), str(sed), tp1), "wb") as f:
        pk.dump(pe13, f)

    with open("./{}/embed2-{}-{}-12-13-{}-0.9.pkl".format(tp, str(ii), str(sed), tp1), "wb") as f:
        pk.dump(pe23, f)

    with open("./{}/post-{}-{}-12-13-{}-0.9.pkl".format(tp, str(ii), str(sed), tp1), "wb") as f:
        pk.dump(pp3, f)

