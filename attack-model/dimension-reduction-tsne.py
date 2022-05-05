from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap


# import torch
# import torch.nn.functional as F
# import torch.optim as optim
#
# from utils import load_data_pokec, accuracy
# from models import GCN_pia
#
# from sklearn.metrics import roc_auc_score, recall_score, precision_score
# from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
#
# import pandas as pd
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    # print('dataMat:', dataMat)
    # print(np.shape(dataMat))
    #print np.array(dataMat)
    #embeddings = np.array(dataMat)
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
    # print('dataMat:', dataMat)
    # print(np.shape(dataMat))
    #print np.array(dataMat)
    #embeddings = np.array(dataMat)
    embeddings = np.array(dataMat)
    # dif = np.zeros(np.shape(embeddings)[0])
    # for i in range(np.shape(embeddings)[1]):
    #     for j in range(i + 1, (np.shape(embeddings)[1])):
    #         dif += np.absolute(embeddings[:, i] - embeddings[:, j])
    # dif = dif / (0.5 * np.shape(embeddings)[1] * (np.shape(embeddings)[1] - 1))
    # embeddings5 = dif
    # embeddings1 = np.mean(embeddings, axis=0)
    # embeddings2 = np.mean(embeddings, axis=1)
    # embeddings3 = np.amax(embeddings, axis=0)
    # embeddings4 = np.amax(embeddings, axis=1)
    return (embeddings)


def plot_embedding_3D(data,label,title):
    # x_min,x_max=np.min(data,axis=0),np.max(data,axis=0)
    # data=(data-x_min)/(x_max-x_min)
    ax=plt.figure()
    ax=Axes3D(ax)

    ax.scatter(data[:,0],data[:,1],data[:,2],c=plt.cm.Set1(label_))
    plt.show()
    return ax


def plot_embedding_2D(data,label,title):
    x_min,x_max=np.min(data,axis=0),np.max(data,axis=0)
    data=(data-x_min)/(x_max-x_min)

    data
    ax=plt.figure()
    colors=plt.cm.rainbow(np.linspace(0,1,2))
    # colors = ListedColormap(['r','g'])

    print(data.shape[0])

    for i in range(data.shape[0]):

        ax.text(data[i,0],data[i,1],'.',color=colors[label[i]],fontdict={'weight':'bold','size':50})
    # plt.legend()
    plt.show()
    plt.savefig('pokec-embed-distribution-nogender' + '.pdf')
    return ax

dataset=['pokec','fb','pubmed']
# tps=['edu-gender','pubmed','gender']
# tps=['edu-gender']
# tps=['fb-small-3.16','pokec-small-3.16','pubmed-small-3.16','fb-interintra-small-3.16']
tps=['pokec','pubmed','fb','fb-interintra','pubmed-interintra','pokec-interintra']
feats={}
feats_neg={}
i=0
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

    label_=[1]*500+[0]*500
    print(len(label_))


    model = TSNE(n_components=2,init='pca',metric='cosine',random_state=0)
    result_3D = model.fit_transform(embed1)
    # print(np.shape(result_3D))

    with open("./tsne-cosine/%s-embed1-tsne-pooling-cosine.pkl"%(tp1), "wb") as f:
        pickle.dump(result_3D, f)

    model = TSNE(n_components=2, init='pca',metric='cosine', random_state=0)
    result_3D = model.fit_transform(embed2)
    print(np.shape(result_3D))

    with open("./tsne-cosine/%s-embed2-tsne-pooling-cosine.pkl"%(tp1), "wb") as f:
        pickle.dump(result_3D, f)

    model = TSNE(n_components=2, init='pca',metric='cosine', random_state=0)
    result_3D = model.fit_transform(posts)
    print(np.shape(result_3D))

    with open("./tsne-cosine/%s-posts-tsne-pooling-cosine.pkl"%(tp1), "wb") as f:
        pickle.dump(result_3D, f)


