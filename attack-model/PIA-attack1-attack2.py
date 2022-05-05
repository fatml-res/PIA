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
    embeddings=np.array(dataMat)

    embeddings1 = np.array(dataMat) ##used for concatenation 
    embeddings2 = np.mean(embeddings, axis=1)  #mean-pooling for each node embedding
    embeddings3 = np.amax(embeddings, axis=1)  #max-pooling for each node embedding


    return embeddings1,embeddings2,embeddings3

def readpost(file_name):
    file = open(file_name)

    dataMat = []
    for line in file.readlines():
        curLine = line.strip().split('\t')
        floatLine = list(map(float, curLine))
        # print(floatLine)
        dataMat.append(floatLine)

    posts = np.array(dataMat) #used for concatenation
    embeddings=posts
    dif = np.zeros(np.shape(embeddings)[0])
    for i in range(np.shape(embeddings)[1]):
        for j in range(i + 1, (np.shape(embeddings)[1])):
            dif += np.absolute(embeddings[:, i] - embeddings[:, j])
    dif = dif / (0.5 * np.shape(embeddings)[1] * (np.shape(embeddings)[1] - 1))
    posts1= dif # element-wise difference
    

    return posts, posts1



tps=['fb','pokec','pubmed','pokec-link','pubmed-link','fb-link']
for tp in tps:


    label=[]
    label_neg=[]

    feat_embed11_neg=[]
    feat_embed12_neg = []
    feat_embed13_neg = []

    feat_embed11=[]
    feat_embed12 = []
    feat_embed13 = []

    feat_embed21_neg = []
    feat_embed22_neg = []
    feat_embed23_neg = []

    feat_embed21 = []
    feat_embed22 = []
    feat_embed23 = []

    feat_post1_neg = []
    feat_post2_neg = []

    feat_post1 = []
    feat_post2 = []

    s=0

    for sed in range(1,6):
        for ii in range(-100,100):  ## totally 1000 subgraphs, ii \in (-100,0): the subgraphs are without property, ii \in (0,100): the subgraphs are with property
            seed=sed

            f = tp

            file1='./%s/embed1-%s-%s-12-13-%s.txt' % (f,ii, seed,tp)
            embed11,embed12,embed13=readembds(file1)
           
            file2='./%s/embed2-%s-%s-12-13-%s.txt' % (f,ii, seed,tp)
            embed21,embed22,embed23=readembds(file2)

            file3='./%s/output_test-%s-%s-12-13-%s.txt' % (f,ii, seed,tp)
            posterior1,posterior2=readpost(file3)
            if (int(s / 100)) % 2 == 0:
            	feat_embed11_neg.append(np.array(embed11).flatten())
                feat_embed12_neg.append(np.array(embed12).flatten())
                feat_embed13_neg.append(np.array(embed13).flatten())
                feat_embed21_neg.append(np.array(embed21).flatten())
                feat_embed22_neg.append(np.array(embed22).flatten())
                feat_embed23_neg.append(np.array(embed23).flatten())
                feat_post1_neg.append(np.array(posterior1).flatten())
                feat_post2_neg.append(np.array(posterior2).flatten())

            else:
            	feat_embed11.append(np.array(embed11).flatten())
                feat_embed12.append(np.array(embed12).flatten())
                feat_embed13.append(np.array(embed13).flatten())
                feat_embed21.append(np.array(embed21).flatten())
                feat_embed22.append(np.array(embed22).flatten())
                feat_embed23.append(np.array(embed23).flatten())
                feat_post1.append(np.array(posterior1).flatten())
                feat_post2.append(np.array(posterior2).flatten())
            s+=1

    for i in range(100):
        label_neg.append([i,0])
    for i in range(100,200):
        label.append([i, 1])
    for i in range(200,300):
        label_neg.append([i,0])
    for i in range(300,400):
        label.append([i, 1])
    for i in range(400,500):
        label_neg.append([i,0])
    for i in range(500,600):
        label.append([i, 1])
    for i in range(600,700):
        label_neg.append([i,0])
    for i in range(700,800):
        label.append([i, 1])
    for i in range(800,900):
        label_neg.append([i,0])
    for i in range(900,1000):
        label.append([i, 1])



    feats=dict()

    feats[11]=np.array(feat_embed11)
    feats[12] = np.array(feat_embed12)
    feats[13] = np.array(feat_embed13)

    feats[21]=np.array(feat_embed21)
    feats[22] = np.array(feat_embed22)
    feats[23] = np.array(feat_embed23)

    feats[1]=np.array(feat_post1)
    feats[2] = np.array(feat_post2)


    feats_neg=dict()

    feats_neg[11]=np.array(feat_embed11_neg)
    feats_neg[12] = np.array(feat_embed12_neg)
    feats_neg[13] = np.array(feat_embed13_neg)

    feats_neg[21]=np.array(feat_embed21_neg)
    feats_neg[22] = np.array(feat_embed22_neg)
    feats_neg[23] = np.array(feat_embed23_neg)

    feats_neg[1]=np.array(feat_post1_neg)
    feats_neg[2] = np.array(feat_post2_neg)

    ft_name=['feat_embed11','feat_embed12','feat_embed13','feat_embed21','feat_embed22','feat_embed23','feat_post1','feat_post2']
    from sklearn.model_selection import train_test_split

    results=[]
    k=[11,12,13,21,22,23,1,2]
    for j in range(len(k)):
        ft_=feats[k[j]]
        ft_neg = feats_neg[k[j]]

        na=ft_name[j]

        print(np.shape(ft_))

        x_train_pos, x_test_pos, y_train_pos, y_test_pos  = train_test_split(ft_, label,test_size=0.3, random_state=42)
        x_train_neg, x_test_neg, y_train_neg, y_test_neg = train_test_split(ft_neg, label_neg, test_size=0.3, random_state=42)

        x_train = np.concatenate((x_train_pos, x_train_neg), axis=0)
        x_test = np.concatenate((x_test_pos, x_test_neg), axis=0)
        y_train = np.concatenate((y_train_pos,y_train_neg), axis=0)
        y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)

        #
        # # ######################################################################

        res_dir='./pia-results/'

        from sklearn import metrics
        from sklearn.neural_network import MLPClassifier

        mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,max_iter=1000)

        print(x_train,y_train[:,1])
        print(np.shape(x_train), np.shape(y_train[:, 1]))
        mlp.fit(x_train, y_train[:,1])

        print("Training set score: %f" % mlp.score(x_train, y_train[:,1]))
        print("Test set score: %f" % mlp.score(x_test, y_test[:,1]))

        y_score = mlp.predict(x_test)
        proba = mlp.predict_proba(x_test)
        proba=np.amax(proba,axis=1)
        print(metrics.f1_score(y_test[:,1], y_score, average='micro'))
        print(metrics.classification_report(y_test[:,1], y_score, labels=range(3)))


        acc = accuracy_score(y_test[:,1],y_score)
        recall = recall_score(y_test[:,1],y_score)
        precision = precision_score(y_test[:,1],y_score)
        f1 = f1_score(y_test[:,1],y_score)
        auc = roc_auc_score(y_test[:,1], proba)


        tsts = []
        for i in range(len(y_score)):
            id_=y_test[i][0]
            prob = proba[i]

            tst = [y_score[i], y_test[i][1],prob, y_test[i][0]]
            tsts.append(tst)
        name = ['y_score', 'y_test_grd','prob', 'index']
        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}/{}-mlp-{}.csv".format(res_dir,na,tp))
        print(acc,recall,precision,f1,auc)

        results.append([acc,recall,precision,f1,auc])

        # # ######################################################################
        # # # ######################################################################

        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(random_state=0)
        lr.fit(x_train, y_train[:,1])


        print("Training set score: %f" % lr.score(x_train, y_train[:,1]))
        print("Test set score: %f" % lr.score(x_test, y_test[:,1]))

        y_score = lr.predict(x_test)
        proba = lr.predict_proba(x_test)
        proba = np.amax(proba, axis=1)
        print(metrics.f1_score(y_test[:,1], y_score, average='micro'))
        print(metrics.classification_report(y_test[:,1], y_score, labels=range(3)))

        acc = accuracy_score(y_test[:,1],y_score)
        recall = recall_score(y_test[:,1],y_score)
        precision = precision_score(y_test[:,1],y_score)
        f1 = f1_score(y_test[:,1],y_score)
        auc = roc_auc_score(y_test[:,1], proba)


        tsts = []
        for i in range(len(y_score)):
            id_=y_test[i][0]
            prob=proba[i]

            tst = [y_score[i], y_test[i][1], prob,y_test[i][0]]
            tsts.append(tst)
        name = ['pred_label', 'grd','prob', 'index']
        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}/{}-lr-{}.csv".format(res_dir,na,tp))
        print(acc, recall, precision, f1, auc)

        results.append([acc, recall, precision, f1, auc])

        # # ######################################################################
        # # # ######################################################################

        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(max_depth=150, random_state=0)
        rf.fit(x_train, y_train[:,1])

        print("Training set score: %f" % rf.score(x_train, y_train[:,1]))
        print("Test set score: %f" % rf.score(x_test, y_test[:,1]))

        y_score = rf.predict(x_test)
        proba = rf.predict_proba(x_test)
        proba = np.amax(proba, axis=1)
        print(metrics.f1_score(y_test[:,1], y_score, average='micro'))
        print(metrics.classification_report(y_test[:,1], y_score, labels=range(3)))


        acc = accuracy_score(y_test[:,1],y_score)
        recall = recall_score(y_test[:,1],y_score)
        precision = precision_score(y_test[:,1],y_score)
        f1 = f1_score(y_test[:,1],y_score)
        auc = roc_auc_score(y_test[:,1], proba)


        tsts = []
        for i in range(len(y_score)):
            id_=y_test[i][0]
            prob=proba[i]

            tst = [y_score[i], y_test[i][1], prob,y_test[i][0]]
            tsts.append(tst)
        name = ['pred_label', 'grd','prob', 'index']
        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}/{}-rf-{}.csv".format(res_dir,na,tp))
        print(acc, recall, precision, f1, auc)

        results.append([acc, recall, precision, f1, auc])

    print(results)
    name = ['acc', 'recall', 'precision', 'f1', 'auc']
    result = pd.DataFrame(columns=name, data=results)
    result.to_csv("{}/results_{}.csv".format(res_dir,tp))
