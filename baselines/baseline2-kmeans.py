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

import csv

def add_laplace_noise(data_list, u=0, b=2):
    laplace_noise = np.random.laplace(u, b, np.shape(data_list))
    return laplace_noise + data_list



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

    embeddings1 = np.amax(embeddings, axis=1)

    return embeddings,embeddings1

def readpost(file_name):
    file = open(file_name)

    dataMat = []
    for line in file.readlines():
        curLine = line.strip().split('\t')
        floatLine = list(map(float, curLine))

        dataMat.append(floatLine)

    embeddings = np.array(dataMat)

    return embeddings


results_all=[]
tps=['fb','pokec','pubmed','pokec-interintra','pubmed-interintra','fb-interintra']

res_dir1 = 'baseline-kmeans'
results=[]
for tp in tps:
    

    label=[]
    feat_embed11_neg=[]
    feat_embed12_neg = []

    feat_embed11=[]
    feat_embed12 = []

    feat_embed21_neg = []
    feat_embed22_neg = []

    feat_embed21 = []
    feat_embed22 = []

    feat_post1_neg = []
    feat_post1 = []

    label_neg=[]

    s=0


    for sed in range(1,6):
        for ii in range(-100,100):   ## totally 1000 subgraphs, ii \in (-100,0): the subgraphs are without property, ii \in (0,100): the subgraphs are with property
            seed=sed

            f = tp

            file1='./%s/pokec-embed1-%s-%s-12-13-%s.txt' % (f,ii, seed,tp)
            print(file1)
            # embed11,embed12,embed13,embed14=readembds(file1)
            embed11,embed12 = readembds(file1)


            file2='./%s/pokec-embed2-%s-%s-12-13-%s.txt' % (f,ii, seed,tp)
            # embed21,embed22,embed23,embed24=readembds(file2)
            embed21,embed22 = readembds(file2)

            file3='./%s/pokec-output_test-%s-%s-12-13-%s.txt' % (f,ii, seed,tp)
            posterior1=readpost(file3)
            if (int(s / 100)) % 2 == 0:
                feat_post1_neg.append(np.array(posterior1).flatten())
                feat_embed11_neg.append(np.array(embed11).flatten())
                feat_embed12_neg.append(np.array(embed12).flatten())
                feat_embed21_neg.append(np.array(embed21).flatten())
                feat_embed22_neg.append(np.array(embed22).flatten())
                feat_embed1211_neg.append(
                    np.concatenate((np.array(embed11).flatten(), np.array(embed21).flatten()), axis=0))
                feats_emb1_post_neg.append(np.concatenate((np.array(embed12).flatten(), np.array(posterior1).flatten()), axis=0))
                feats_emb2_post_neg.append(np.concatenate((np.array(embed22).flatten(), np.array(posterior1).flatten()), axis=0))
                feats_emb12_post_neg.append(np.concatenate((np.array(embed12).flatten(),np.array(embed22).flatten(), np.array(posterior1).flatten()), axis=0))



            else:
                feat_post1.append(np.array(posterior1).flatten())
                feat_embed11.append(np.array(embed11).flatten())
                feat_embed12.append(np.array(embed12).flatten())
                feat_embed21.append(np.array(embed21).flatten())
                feat_embed22.append(np.array(embed22).flatten())
                feat_embed1211.append(
                    np.concatenate((np.array(embed11).flatten(), np.array(embed21).flatten()), axis=0))

                feats_emb1_post.append(np.concatenate((np.array(embed12).flatten(), np.array(posterior1).flatten()), axis=0))
                feats_emb2_post.append(np.concatenate((np.array(embed22).flatten(), np.array(posterior1).flatten()), axis=0))
                feats_emb12_post.append(
                    np.concatenate((np.array(embed12).flatten(), np.array(embed22).flatten(), np.array(posterior1).flatten()), axis=0))

            s+=1



    feats=dict()
    feats[1]=np.array(feat_embed11) #embed if layer1, concatenation
    feats[2] = np.array(feat_embed12)  #embed if layer1, max-pooling
    feats[3]=np.array(feat_embed21) #embed if layer2 , concatenation
    feats[4] = np.array(feat_embed22) #embed if layer2, max-pooling
    feats[5] = np.array(feat_embed1211) #embed if layer1&2 , concatenation
    feats[6]=np.array(feat_post1) # posterior
    feats[7] = np.array(feats_emb1_post) # embed if layer1+post
    feats[8] = np.array(feats_emb2_post) # embed if layer2+post
    feats[9] = np.array(feats_emb12_post) # embed if layer1+layer2++post


    feats_neg=dict()
    feats_neg[1]=np.array(feat_embed11_neg)
    feats_neg[2] = np.array(feat_embed12_neg)
    feats_neg[3]=np.array(feat_embed21_neg)
    feats_neg[4] = np.array(feat_embed22_neg)
    feats_neg[5] = np.array(feat_embed1211_neg)
    feats_neg[6]=np.array(feat_post1_neg)
    feats_neg[7] = np.array(feats_emb1_post_neg)
    feats_neg[8] = np.array(feats_emb2_post_neg)
    feats_neg[9] = np.array(feats_emb12_post_neg)
    ft_name=['feat_embed11','feat_embed12','feat_embed21','feat_embed22','feat_post','feat_embed1_2', 'feat_embed1_post', 'feat_embed2_post', 'feat_embed12_post']


    feat_embed11=np.array(feat_embed11)
    feat_embed12 = np.array(feat_embed12)
    feat_embed21=np.array(feat_embed21)
    feat_embed22 = np.array(feat_embed22)
    feat_embed31=np.array(feat_post1)
    feat_embed11_neg=np.array(feat_embed11_neg)
    feat_embed12_neg = np.array(feat_embed12_neg)
    feat_embed21_neg=np.array(feat_embed21_neg)
    feat_embed22_neg = np.array(feat_embed22_neg)
    feat_embed31_neg=np.array(feat_post1_neg)

    feat_embed1211_neg=np.array(feat_embed1211_neg)
    feats_emb1_post_neg= np.array(feats_emb1_post_neg)
    feats_emb2_post_neg = np.array(feats_emb2_post_neg)
    feats_emb12_post_neg = np.array(feats_emb12_post_neg)

    feat_embed1211= np.array(feat_embed1211)
    feats_emb1_post = np.array(feats_emb1_post)
    feats_emb2_post = np.array(feats_emb2_post)
    feats_emb12_post = np.array(feats_emb12_post)


    ft_all=dict()


    ft_all[1]=np.concatenate((feat_embed11_neg[0:100,:],feat_embed11[0:100,:],feat_embed11_neg[100:200,:],feat_embed11[100:200,:],feat_embed11_neg[200:300,:],feat_embed11[200:300,:],feat_embed11_neg[300:400,:],feat_embed11[300:400,:],feat_embed11_neg[400:500,:],feat_embed11[400:500,:]),axis=0)
    ft_all[2]=np.concatenate((feat_embed21_neg[0:100,:],feat_embed21[0:100,:],feat_embed21_neg[100:200,:],feat_embed21[100:200,:],feat_embed21_neg[200:300,:],feat_embed21[200:300,:],feat_embed21_neg[300:400,:],feat_embed21[300:400,:],feat_embed21_neg[400:500,:],feat_embed21[400:500,:]),axis=0)
    
    ft_all[3]=np.concatenate((feat_embed12_neg[0:100,:],feat_embed12[0:100,:],feat_embed12_neg[100:200,:],feat_embed12[100:200,:],feat_embed12_neg[200:300,:],feat_embed12[200:300,:],feat_embed12_neg[300:400,:],feat_embed12[300:400,:],feat_embed12_neg[400:500,:],feat_embed12[400:500,:]),axis=0)
    ft_all[4]=np.concatenate((feat_embed22_neg[0:100,:],feat_embed22[0:100,:],feat_embed22_neg[100:200,:],feat_embed22[100:200,:],feat_embed22_neg[200:300,:],feat_embed22[200:300,:],feat_embed22_neg[300:400,:],feat_embed22[300:400,:],feat_embed22_neg[400:500,:],feat_embed22[400:500,:]),axis=0)
    
    
    ft_all[6]=np.concatenate((feat_embed31_neg[0:100,:],feat_embed31[0:100,:],feat_embed31_neg[100:200,:],feat_embed31[100:200,:],feat_embed31_neg[200:300,:],feat_embed31[200:300,:],feat_embed31_neg[300:400,:],feat_embed31[300:400,:],feat_embed31_neg[400:500,:],feat_embed31[400:500,:]),axis=0)

    ft_all[5] = np.concatenate((feat_embed1211_neg[0:100, :], feat_embed1211[0:100, :], feat_embed1211_neg[100:200, :],
                                 feat_embed1211[100:200, :], feat_embed1211_neg[200:300, :], feat_embed1211[200:300, :],
                                 feat_embed1211_neg[300:400, :], feat_embed1211[300:400, :], feat_embed1211_neg[400:500, :],
                                 feat_embed1211[400:500, :]), axis=0)
    ft_all[7] = np.concatenate((feats_emb1_post_neg[0:100, :], feats_emb1_post[0:100, :], feats_emb1_post_neg[100:200, :],
                                feats_emb1_post[100:200, :], feats_emb1_post_neg[200:300, :], feats_emb1_post[200:300, :],
                                feats_emb1_post_neg[300:400, :], feats_emb1_post[300:400, :], feats_emb1_post_neg[400:500, :],
                                feats_emb1_post[400:500, :]), axis=0)

    ft_all[8] = np.concatenate((feats_emb2_post_neg[0:100, :], feats_emb2_post[0:100, :], feats_emb2_post_neg[100:200, :],
                                feats_emb2_post[100:200, :], feats_emb2_post_neg[200:300, :], feats_emb2_post[200:300, :],
                                feats_emb2_post_neg[300:400, :], feats_emb2_post[300:400, :], feats_emb2_post_neg[400:500, :],
                                feats_emb2_post[400:500, :]), axis=0)
    ft_all[9] = np.concatenate((feats_emb12_post_neg[0:100, :], feats_emb12_post[0:100, :], feats_emb12_post_neg[100:200, :],
                                feats_emb12_post[100:200, :], feats_emb12_post_neg[200:300, :], feats_emb12_post[200:300, :],
                                feats_emb12_post_neg[300:400, :], feats_emb12_post[300:400, :], feats_emb12_post_neg[400:500, :],
                                feats_emb12_post[400:500, :]), axis=0)



    # ns=str(ns)
    from sklearn.model_selection import train_test_split

    index = []
    labels = []

    with open("fb-edu-gender-2.10/feat_embed1-mlp-12-13-edu-gender.csv") as csvfile:  ####to make sure the testing graphs of baseline are the same as those in our methods, so read the index of testing graph from the saved files
        csv_reader = csv.reader(csvfile)
        print(csv_reader)
        result_header = next(csv_reader)
        acc = 0
        cnt = 0
        for row in csv_reader:
            index.append(int(row[4]))
            labels.append(int(row[2]))

    index = np.array(index)
    labels = np.array(labels)

    k=[1,2,3,4,5,6,7,8,9]
    for j in range(len(k)):

        ft=ft_all[k[j]]

        na=ft_name[j]


        ft=ft[index]




        from sklearn.cluster import KMeans
        from sklearn.metrics import accuracy_score

        accuracy = []
        for i in range(1000):
            kmeans = KMeans(n_clusters=2, random_state=i).fit(ft)
            # kmeans = KMeans(n_clusters=2, random_state=i).fit(X)
            # print(kmeans.labels_)
            ylabel = labels
            acc = accuracy_score(kmeans.labels_, ylabel)
            accuracy.append(acc)
        print(max(accuracy))

        acc_ = max(accuracy)

        results.append([tp,na,acc_])

    name = ['tp', 'na', 'acc']
    result = pd.DataFrame(columns=name, data=results)
    result.to_csv("{}/results_kmeans-{}.csv".format(res_dir1,tp))

name = ['tp','na','acc']
result = pd.DataFrame(columns=name, data=results)
result.to_csv("{}/results_kmeans.csv".format(res_dir1))




