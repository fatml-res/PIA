from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
# import torch.nn.functional as F
# import torch.optim as optim
#
# from utils import load_data_pokec, accuracy
# from models import GCN_pia

from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import pandas as pd
import pickle as pk

# from keras.layers import Input, Dense
from scipy.special import softmax




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

    embedding_=[]
    for ii in embeddings:
        embedding_.append(softmax(ii))  ##softmax normalization
    embed1=np.array(embedding_)


    embeddings4 = np.amax(embed1, axis=1)


    return embeddings4,embed1

def readpost(file_name):
    file = open(file_name)

    dataMat = []
    for line in file.readlines():
        curLine = line.strip().split('\t')
        floatLine = list(map(float, curLine))
        # print(floatLine)
        dataMat.append(floatLine)

    embeddings = np.array(dataMat)

    embedding_ = []
    for ii in embeddings:
        embedding_.append(softmax(ii))
    embed11 = np.array(embedding_)

    return embed11



tps=['fb','pokec','pubmed','pokec-interintra','pubmed-interintra','fb-interintra']
for tp in tps:

    label=[]
    feat_embed11_neg=[]
    feat_embed12_neg = []
    feat_embed13_neg = []
    feat_embed14_neg = []

    feat_embed11=[]
    feat_embed12 = []
    feat_embed13 = []
    feat_embed14 = []

    feat_embed21_neg = []
    feat_embed22_neg = []
    feat_embed23_neg = []
    feat_embed24_neg = []

    feat_embed21 = []
    feat_embed22 = []
    feat_embed23 = []
    feat_embed24 = []

    feat_post1_neg = []
    feat_post2_neg = []
    feat_post3_neg = []
    feat_post4_neg = []
    feat_post5_neg = []

    feat_post1 = []
    feat_post2 = []
    feat_post3 = []
    feat_post4 = []
    feat_post5 = []

    feat_embed1121_neg=[]
    feat_embed1121=[]

    label_neg=[]

    s=0

    for sed in range(1,6):
        for ii in range(-100,100):
            seed=sed

            f = tp

            file1='./%s/embed1-%s-%s-12-13-%s.txt' % (f,ii, seed,tp)
            print(file1)
            embed11,embed12=readembds(file1)

            file2='./%s/pokec-embed2-%s-%s-12-13-%s.txt' % (f,ii, seed,tp)
            embed21, embed22=readembds(file2)

            file3='./%s/pokec-output_test-%s-%s-12-13-%s.txt' % (f,ii, seed,tp)
            posterior1,posterior2=readpost(file3)


            if (int(s / 100)) % 2 == 0:
                feat_post1_neg.append(np.array(posterior1).flatten())
                feat_post2_neg.append(np.array(posterior2).flatten())
                feat_embed11_neg.append(np.array(embed11).flatten())
                feat_embed12_neg.append(np.array(embed12).flatten())
                feat_embed21_neg.append(np.array(embed21).flatten())
                feat_embed22_neg.append(np.array(embed22).flatten())


            else:
                feat_post1.append(np.array(posterior1).flatten())
                feat_post2_neg.append(np.array(posterior2).flatten())
                feat_embed11.append(np.array(embed11).flatten())
                feat_embed12.append(np.array(embed12).flatten())
                feat_embed21.append(np.array(embed21).flatten())
                feat_embed22.append(np.array(embed22).flatten())

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

    feats[1]=np.array(feat_embed11)
    feats[2] = np.array(feat_embed12)

    feats[3]=np.array(feat_embed21)
    feats[4] = np.array(feat_embed22)

    feats[5]=np.array(feat_post1)

    feats_neg=dict()

    feats_neg[1]=np.array(feat_embed11_neg)
    feats_neg[2] = np.array(feat_embed12_neg)

    feats_neg[3]=np.array(feat_embed21_neg)
    feats_neg[4] = np.array(feat_embed22_neg)

    feats_neg[5]=np.array(feat_post1_neg)

    ft_name=['feat_embed11','feat_embed12','feat_embed21','feat_embed22','feat_post1']

    from sklearn.model_selection import train_test_split


    results=[]
    k=[1,2,3,4,5]
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

        # X_train, X_test, y_train, y_test = train_test_split(sim_matrix, y_label, test_size=0.3, random_state=42)
        #
        # # ######################################################################

        res_dir='./defense-norm/'

        from sklearn import metrics
        from sklearn.neural_network import MLPClassifier

        mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,max_iter=1500,early_stopping=True)

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
        # report = metrics.classification_report(y_test[:,1], y_score, labels=range(3), output_dict=True)
        #
        # out = "{}/{}-mlp-report-12-13-{}.csv".format(res_dir, na,tp)
        #
        # df = pd.DataFrame(report).transpose()
        # df.to_csv(out, index=True)

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

        results.append([na,acc,recall,precision,f1,auc])

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

        # report=metrics.classification_report(y_test[:,1], y_score, labels=range(3),output_dict=True)
        # # print(report)
        #
        # out = "{}/{}-lr-report-12-13-{}.csv".format(res_dir, na,tp)
        #
        # df = pd.DataFrame(report).transpose()
        # df.to_csv(out, index=True)

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

        results.append([na,acc, recall, precision, f1, auc])

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

        # report=metrics.classification_report(y_test[:,1], y_score, labels=range(3),output_dict=True)
        # # print(report)
        #
        # out = "{}/{}-rf-report-12-13-{}.csv".format(res_dir, na,tp)
        #
        # df = pd.DataFrame(report).transpose()
        # df.to_csv(out, index=True)

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

        results.append([na,acc, recall, precision, f1, auc])

    print(results)
    name = ['tp','acc', 'recall', 'precision', 'f1', 'auc']
    result = pd.DataFrame(columns=name, data=results)
    result.to_csv("{}/results_{}-norm-softmax.csv".format(res_dir,tp))