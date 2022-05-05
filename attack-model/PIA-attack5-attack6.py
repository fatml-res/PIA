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

import random

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
    # embeddings1=np.mean(embeddings,axis=0)
    # embeddings2 = np.mean(embeddings, axis=1)
    # embeddings3 = np.amax(embeddings, axis=0)
    embeddings4 = np.amax(embeddings, axis=1)
    return embeddings4

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
    return embeddings


# s=0
dataset=['pokec','fb','pubmed']
tps=['pokec-interintra-small-3.16','fb-interintra-small-3.16','pubmed-interintra-small-3.16']
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
            f_fir=tp1

            
            file1 = './%s/embed1-%s-%s-12-13-%s.txt' % (f_fir, ii, seed, tp1)
            embed1 = readembds(file1)

            file2 = './%s/embed2-%s-%s-12-13-%s.txt' % (f_fir, ii, seed, tp1)
            embed2 = readembds(file2)

            file3 = './%s/output_test-%s-%s-12-13-%s.txt' % (f_fir, ii, seed, tp1)

            posterior = readpost(file3)


            if (int(s / 100)) % 2 == 0:
                feat_embed1_neg.append(np.array(embed1).flatten())
                feat_embed2_neg.append(np.array(embed2).flatten())
                feat_embed12_neg.append(
                    np.concatenate((np.array(embed1).flatten(), (np.array(embed2).flatten())), axis=0))
                feat_post_neg.append(np.array(posterior).flatten())
            else:
                feat_embed1.append(np.array(embed1).flatten())
                feat_embed2.append(np.array(embed2).flatten())

                feat_embed12.append(np.concatenate((np.array(embed1).flatten(), (np.array(embed2).flatten())), axis=0))

                feat_post.append(np.array(posterior).flatten())


    feats[tp1]=[np.array(feat_embed1),np.array(feat_embed2),np.array(feat_embed12),np.array(feat_post)]
    feats_neg[tp1] = [np.array(feat_embed1_neg),np.array(feat_embed2_neg),np.array(feat_embed12_neg),np.array(feat_post_neg)]
    i+=1




for i in range(100):
    label_neg.append([i,0])
for i in range(100,200):
    label.append([i, 1])
for i in range(200,300):
    label_neg.append([i,0],)
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

ft_name=['feat_embed1','feat_embed2','feat_embed12','feat_post']
from sklearn.model_selection import train_test_split

results=[]
# k=[1,4,6]
# k = [0,1,3]
k = [2]
tps=['pokec-interintra','fb-interintra','pubmed-interintra']
train_tests={}
for j in k:
    for tp1 in tps:

        train_test=[]

        ft_ = feats[tp1][j]
        ft_neg = feats_neg[tp1][j]

        x_train_pos, x_test_pos, y_train_pos, y_test_pos = train_test_split(ft_, label, test_size=0.2, random_state=42)
        x_train_neg, x_test_neg, y_train_neg, y_test_neg = train_test_split(ft_neg, label_neg, test_size=0.2,
                                                                            random_state=42)

        # x_train = np.concatenate((x_train_pos, x_train_neg), axis=0)
        # x_test = np.concatenate((x_test_pos, x_test_neg), axis=0)
        # y_train = np.concatenate((y_train_pos, y_train_neg), axis=0)
        # y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)

        train_test.append(np.array(x_train_pos))
        train_test.append(np.array(x_test_pos))
        train_test.append(np.array(y_train_pos))
        train_test.append(np.array(y_test_pos))

        train_test.append(np.array(x_train_neg))
        train_test.append(np.array(x_test_neg))
        train_test.append(np.array(y_train_neg))
        train_test.append(np.array(y_test_neg))


        # xx_train[tp1]=x_train
        # xx_test[tp1]=x_test
        # yy_train[tp1]=y_train
        # yy_test[tp1]=y_test
        train_test=np.array(train_test)
        tp_na=str(j)+tp1
        train_tests[tp_na]=train_test

ratios=[0.1,0.25,0.5,1]

for iii in range(len(tps)):

    for ii in range(iii+1,len(tps)):

        for j in k:
            for r in ratios:
                tp1 = tps[iii]
                tp2 = tps[ii]
                tp_na1 = str(j) + tp1
                train_test1=np.array(train_tests[tp_na1])

                index_all = list(range(400))
                num_sample=int(r*400)

                index = np.array(random.sample(index_all, num_sample))

                x_train_pos1=train_test1[0][index]
                x_test_pos1=train_test1[1]
                y_train_pos1=train_test1[2][index]
                y_test_pos1=train_test1[3]
                x_train_neg1=train_test1[4][index]
                x_test_neg1=train_test1[5]
                y_train_neg1=train_test1[6][index]
                y_test_neg1=train_test1[7]

                tp_na2 = str(j) + tp2
                train_test2=train_tests[tp_na2]

                x_train_pos2=train_test2[0]
                x_test_pos2=train_test2[1]
                y_train_pos2=train_test2[2]
                y_test_pos2=train_test2[3]
                x_train_neg2=train_test2[4]
                x_test_neg2=train_test2[5]
                y_train_neg2=train_test2[6]
                y_test_neg2=train_test2[7]



                x_train = np.concatenate((x_train_pos1, x_train_neg1,x_train_pos2, x_train_neg2), axis=0)

                y_train = np.concatenate((y_train_pos1, y_train_neg1,y_train_pos2, y_train_neg2), axis=0)

                x_test1 = np.concatenate((x_test_pos1, x_test_neg1), axis=0)
                y_test1 = np.concatenate((y_test_pos1, y_test_neg1), axis=0)

                x_test2 = np.concatenate((x_test_pos2, x_test_neg2), axis=0)
                y_test2 = np.concatenate((y_test_pos2, y_test_neg2), axis=0)

                # # ######################################################################

                res_dir = './attack56/'

                from sklearn import metrics
                from sklearn.neural_network import MLPClassifier

                mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,
                                    max_iter=1500,early_stopping=True)

                # print(x_train,y_train[:,1])
                # print(np.shape(x_train), np.shape(y_train[:, 1]))
                mlp.fit(x_train, y_train[:, 1])

                print("Training set score: %f" % mlp.score(x_train, y_train[:, 1]))
                print("Test set score: %f" % mlp.score(x_test1, y_test1[:, 1]))

                y_score = mlp.predict(x_test1)
                proba = mlp.predict_proba(x_test1)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test1[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test1[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test1[:, 1], y_score)
                recall = recall_score(y_test1[:, 1], y_score)
                precision = precision_score(y_test1[:, 1], y_score)
                f1 = f1_score(y_test1[:, 1], y_score)
                auc = roc_auc_score(y_test1[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test1[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test1[i][1], prob, y_test1[i][0]]
                    tsts.append(tst)
                na = tp1 + '-' + tp2 + '-' + str(j)+'-' + str(r)
                name = ['y_score', 'y_test_grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-mlp-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])





                y_score = mlp.predict(x_test2)
                proba = mlp.predict_proba(x_test2)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test2[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test2[:, 1], y_score, labels=range(3)))


                acc = accuracy_score(y_test2[:, 1], y_score)
                recall = recall_score(y_test2[:, 1], y_score)
                precision = precision_score(y_test2[:, 1], y_score)
                f1 = f1_score(y_test2[:, 1], y_score)
                auc = roc_auc_score(y_test2[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test2[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test2[i][1], prob, y_test2[i][0]]
                    tsts.append(tst)
                na = tp2 + '-' + tp1 + '-' + str(j)+'-' + str(r)
                name = ['y_score', 'y_test_grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-mlp-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])




                # # ######################################################################
                # # # ######################################################################

                from sklearn.linear_model import LogisticRegression

                lr = LogisticRegression(random_state=0)
                lr.fit(x_train, y_train[:, 1])

                print("Training set score: %f" % lr.score(x_train, y_train[:, 1]))
                print("Test set score: %f" % lr.score(x_test1, y_test1[:, 1]))

                y_score = lr.predict(x_test1)
                proba = lr.predict_proba(x_test1)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test1[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test1[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test1[:, 1], y_score)
                recall = recall_score(y_test1[:, 1], y_score)
                precision = precision_score(y_test1[:, 1], y_score)
                f1 = f1_score(y_test1[:, 1], y_score)
                auc = roc_auc_score(y_test1[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test1[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test1[i][1], prob, y_test1[i][0]]
                    tsts.append(tst)
                na = tp1 + '-' + tp2 + '-' + str(j)+'-' + str(r)
                name = ['pred_label', 'grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-lr-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])



                y_score = lr.predict(x_test2)
                proba = lr.predict_proba(x_test2)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test2[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test2[:, 1], y_score, labels=range(3)))


                acc = accuracy_score(y_test2[:, 1], y_score)
                recall = recall_score(y_test2[:, 1], y_score)
                precision = precision_score(y_test2[:, 1], y_score)
                f1 = f1_score(y_test2[:, 1], y_score)
                auc = roc_auc_score(y_test2[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test2[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test2[i][1], prob, y_test2[i][0]]
                    tsts.append(tst)
                na = tp2 + '-' + tp1 + '-' + str(j)+'-' + str(r)
                name = ['pred_label', 'grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-lr-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])

                # # ######################################################################
                # # # ######################################################################

                from sklearn.ensemble import RandomForestClassifier

                rf = RandomForestClassifier(max_depth=150, random_state=0)
                rf.fit(x_train, y_train[:, 1])

                print("Training set score: %f" % rf.score(x_train, y_train[:, 1]))
                print("Test set score: %f" % rf.score(x_test1, y_test1[:, 1]))

                y_score = rf.predict(x_test1)
                proba = rf.predict_proba(x_test1)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test1[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test1[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test1[:, 1], y_score)
                recall = recall_score(y_test1[:, 1], y_score)
                precision = precision_score(y_test1[:, 1], y_score)
                f1 = f1_score(y_test1[:, 1], y_score)
                auc = roc_auc_score(y_test1[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test1[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test1[i][1], prob, y_test1[i][0]]
                    tsts.append(tst)
                na = tp1 + '-' + tp2 + '-' + str(j)+'-' + str(r)
                name = ['pred_label', 'grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-rf-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])



                y_score = rf.predict(x_test2)
                proba = rf.predict_proba(x_test2)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test2[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test2[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test2[:, 1], y_score)
                recall = recall_score(y_test2[:, 1], y_score)
                precision = precision_score(y_test2[:, 1], y_score)
                f1 = f1_score(y_test2[:, 1], y_score)
                auc = roc_auc_score(y_test2[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test2[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test2[i][1], prob, y_test2[i][0]]
                    tsts.append(tst)
                na = tp2 + '-' + tp1 + '-' + str(j)+'-' + str(r)
                name = ['pred_label', 'grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-rf-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])



                tp2 = tps[iii]
                tp1 = tps[ii]
                tp_na1 = str(j) + tp1
                train_test1 = np.array(train_tests[tp_na1])

                index_all = list(range(400))
                num_sample = int(r * 400)

                index = np.array(random.sample(index_all, num_sample))
                x_train_pos1=train_test1[0][index]
                x_test_pos1=train_test1[1]
                y_train_pos1=train_test1[2][index]
                y_test_pos1=train_test1[3]
                x_train_neg1=train_test1[4][index]
                x_test_neg1=train_test1[5]
                y_train_neg1=train_test1[6][index]
                y_test_neg1=train_test1[7]

                tp_na2 = str(j) + tp2
                train_test2=train_tests[tp_na2]

                x_train_pos2=train_test2[0]
                x_test_pos2=train_test2[1]
                y_train_pos2=train_test2[2]
                y_test_pos2=train_test2[3]
                x_train_neg2=train_test2[4]
                x_test_neg2=train_test2[5]
                y_train_neg2=train_test2[6]
                y_test_neg2=train_test2[7]



                x_train = np.concatenate((x_train_pos1, x_train_neg1,x_train_pos2, x_train_neg2), axis=0)

                y_train = np.concatenate((y_train_pos1, y_train_neg1,y_train_pos2, y_train_neg2), axis=0)

                x_test1 = np.concatenate((x_test_pos1, x_test_neg1), axis=0)
                y_test1 = np.concatenate((y_test_pos1, y_test_neg1), axis=0)

                x_test2 = np.concatenate((x_test_pos2, x_test_neg2), axis=0)
                y_test2 = np.concatenate((y_test_pos2, y_test_neg2), axis=0)

                # # ######################################################################

                res_dir = './attack56/'

                from sklearn import metrics
                from sklearn.neural_network import MLPClassifier

                mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,
                                    max_iter=1500,early_stopping=True)

                # print(x_train,y_train[:,1])
                # print(np.shape(x_train), np.shape(y_train[:, 1]))
                mlp.fit(x_train, y_train[:, 1])

                print("Training set score: %f" % mlp.score(x_train, y_train[:, 1]))
                print("Test set score: %f" % mlp.score(x_test1, y_test1[:, 1]))

                y_score = mlp.predict(x_test1)
                proba = mlp.predict_proba(x_test1)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test1[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test1[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test1[:, 1], y_score)
                recall = recall_score(y_test1[:, 1], y_score)
                precision = precision_score(y_test1[:, 1], y_score)
                f1 = f1_score(y_test1[:, 1], y_score)
                auc = roc_auc_score(y_test1[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test1[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test1[i][1], prob, y_test1[i][0]]
                    tsts.append(tst)
                na = tp1 + '-' + tp2 + '-' + str(j)+'-' + str(r)
                name = ['y_score', 'y_test_grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-mlp-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])





                y_score = mlp.predict(x_test2)
                proba = mlp.predict_proba(x_test2)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test2[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test2[:, 1], y_score, labels=range(3)))


                acc = accuracy_score(y_test2[:, 1], y_score)
                recall = recall_score(y_test2[:, 1], y_score)
                precision = precision_score(y_test2[:, 1], y_score)
                f1 = f1_score(y_test2[:, 1], y_score)
                auc = roc_auc_score(y_test2[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test2[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test2[i][1], prob, y_test2[i][0]]
                    tsts.append(tst)
                na = tp2 + '-' + tp1 + '-' + str(j)+'-' + str(r)
                name = ['y_score', 'y_test_grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-mlp-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])




                # # ######################################################################
                # # # ######################################################################

                from sklearn.linear_model import LogisticRegression

                lr = LogisticRegression(random_state=0)
                lr.fit(x_train, y_train[:, 1])

                print("Training set score: %f" % lr.score(x_train, y_train[:, 1]))
                print("Test set score: %f" % lr.score(x_test1, y_test1[:, 1]))

                y_score = lr.predict(x_test1)
                proba = lr.predict_proba(x_test1)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test1[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test1[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test1[:, 1], y_score)
                recall = recall_score(y_test1[:, 1], y_score)
                precision = precision_score(y_test1[:, 1], y_score)
                f1 = f1_score(y_test1[:, 1], y_score)
                auc = roc_auc_score(y_test1[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test1[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test1[i][1], prob, y_test1[i][0]]
                    tsts.append(tst)
                na = tp1 + '-' + tp2 + '-' + str(j)+'-' + str(r)
                name = ['pred_label', 'grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-lr-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])



                y_score = lr.predict(x_test2)
                proba = lr.predict_proba(x_test2)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test2[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test2[:, 1], y_score, labels=range(3)))


                acc = accuracy_score(y_test2[:, 1], y_score)
                recall = recall_score(y_test2[:, 1], y_score)
                precision = precision_score(y_test2[:, 1], y_score)
                f1 = f1_score(y_test2[:, 1], y_score)
                auc = roc_auc_score(y_test2[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test2[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test2[i][1], prob, y_test2[i][0]]
                    tsts.append(tst)
                na = tp2 + '-' + tp1 + '-' + str(j)+'-' + str(r)
                name = ['pred_label', 'grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-lr-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])

                # # ######################################################################
                # # # ######################################################################

                from sklearn.ensemble import RandomForestClassifier

                rf = RandomForestClassifier(max_depth=150, random_state=0)
                rf.fit(x_train, y_train[:, 1])

                print("Training set score: %f" % rf.score(x_train, y_train[:, 1]))
                print("Test set score: %f" % rf.score(x_test1, y_test1[:, 1]))

                y_score = rf.predict(x_test1)
                proba = rf.predict_proba(x_test1)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test1[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test1[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test1[:, 1], y_score)
                recall = recall_score(y_test1[:, 1], y_score)
                precision = precision_score(y_test1[:, 1], y_score)
                f1 = f1_score(y_test1[:, 1], y_score)
                auc = roc_auc_score(y_test1[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test1[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test1[i][1], prob, y_test1[i][0]]
                    tsts.append(tst)
                na = tp1 + '-' + tp2 + '-' + str(j)+'-' + str(r)
                name = ['pred_label', 'grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-rf-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])



                y_score = rf.predict(x_test2)
                proba = rf.predict_proba(x_test2)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test2[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test2[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test2[:, 1], y_score)
                recall = recall_score(y_test2[:, 1], y_score)
                precision = precision_score(y_test2[:, 1], y_score)
                f1 = f1_score(y_test2[:, 1], y_score)
                auc = roc_auc_score(y_test2[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test2[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test2[i][1], prob, y_test2[i][0]]
                    tsts.append(tst)
                na = tp2 + '-' + tp1 + '-' + str(j)+'-' + str(r)
                name = ['pred_label', 'grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-rf-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])






print(results)

name = ['name','acc', 'recall', 'precision', 'f1', 'auc']
result = pd.DataFrame(columns=name, data=results)
result.to_csv("{}/results_attack56-interintra.csv".format(res_dir))

tps = ['pokec', 'fb', 'pubmed']
feats = {}
feats_neg = {}
i = 0
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
            f_fir = tp1

            file1 = './%s/embed1-%s-%s-12-13-%s.txt' % (f_fir, ii, seed, tp1)
            embed1 = readembds(file1)

            file2 = './%s/embed2-%s-%s-12-13-%s.txt' % (f_fir, ii, seed, tp1)
            embed2 = readembds(file2)

            file3 = './%s/output_test-%s-%s-12-13-%s.txt' % (f_fir, ii, seed, tp1)

            posterior = readpost(file3)

            if (int(s / 100)) % 2 == 0:
                feat_embed1_neg.append(np.array(embed1).flatten())
                feat_embed2_neg.append(np.array(embed2).flatten())
                feat_embed12_neg.append(
                    np.concatenate((np.array(embed1).flatten(), (np.array(embed2).flatten())), axis=0))
                feat_post_neg.append(np.array(posterior).flatten())
            else:
                feat_embed1.append(np.array(embed1).flatten())
                feat_embed2.append(np.array(embed2).flatten())

                feat_embed12.append(np.concatenate((np.array(embed1).flatten(), (np.array(embed2).flatten())), axis=0))

                feat_post.append(np.array(posterior).flatten())

    feats[tp1] = [np.array(feat_embed1), np.array(feat_embed2), np.array(feat_embed12), np.array(feat_post)]
    feats_neg[tp1] = [np.array(feat_embed1_neg), np.array(feat_embed2_neg), np.array(feat_embed12_neg),
                      np.array(feat_post_neg)]
    i += 1

for i in range(100):
    label_neg.append([i, 0])
for i in range(100, 200):
    label.append([i, 1])
for i in range(200, 300):
    label_neg.append([i, 0], )
for i in range(300, 400):
    label.append([i, 1])
for i in range(400, 500):
    label_neg.append([i, 0])
for i in range(500, 600):
    label.append([i, 1])
for i in range(600, 700):
    label_neg.append([i, 0])
for i in range(700, 800):
    label.append([i, 1])
for i in range(800, 900):
    label_neg.append([i, 0])
for i in range(900, 1000):
    label.append([i, 1])

ft_name = ['feat_embed1', 'feat_embed2', 'feat_embed12', 'feat_post']
from sklearn.model_selection import train_test_split

results = []
# k=[1,4,6]
k = [0,1,2,3]
tps = ['pokec', 'fb', 'pubmed']
train_tests = {}
for j in k:
    for tp1 in tps:
        train_test = []

        ft_ = feats[tp1][j]
        ft_neg = feats_neg[tp1][j]

        x_train_pos, x_test_pos, y_train_pos, y_test_pos = train_test_split(ft_, label, test_size=0.2, random_state=42)
        x_train_neg, x_test_neg, y_train_neg, y_test_neg = train_test_split(ft_neg, label_neg, test_size=0.2,
                                                                            random_state=42)

        # x_train = np.concatenate((x_train_pos, x_train_neg), axis=0)
        # x_test = np.concatenate((x_test_pos, x_test_neg), axis=0)
        # y_train = np.concatenate((y_train_pos, y_train_neg), axis=0)
        # y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)

        train_test.append(np.array(x_train_pos))
        train_test.append(np.array(x_test_pos))
        train_test.append(np.array(y_train_pos))
        train_test.append(np.array(y_test_pos))

        train_test.append(np.array(x_train_neg))
        train_test.append(np.array(x_test_neg))
        train_test.append(np.array(y_train_neg))
        train_test.append(np.array(y_test_neg))

        # xx_train[tp1]=x_train
        # xx_test[tp1]=x_test
        # yy_train[tp1]=y_train
        # yy_test[tp1]=y_test
        train_test = np.array(train_test)
        tp_na = str(j) + tp1
        train_tests[tp_na] = train_test

ratios = [0.1, 0.25, 0.5, 1]

for iii in range(len(tps)):

    for ii in range(iii + 1, len(tps)):

        for j in k:
            for r in ratios:
                tp1 = tps[iii]
                tp2 = tps[ii]
                tp_na1 = str(j) + tp1
                train_test1 = np.array(train_tests[tp_na1])

                index_all = list(range(400))
                num_sample = int(r * 400)

                index = np.array(random.sample(index_all, num_sample))

                x_train_pos1 = train_test1[0][index]
                x_test_pos1 = train_test1[1]
                y_train_pos1 = train_test1[2][index]
                y_test_pos1 = train_test1[3]
                x_train_neg1 = train_test1[4][index]
                x_test_neg1 = train_test1[5]
                y_train_neg1 = train_test1[6][index]
                y_test_neg1 = train_test1[7]

                tp_na2 = str(j) + tp2
                train_test2 = train_tests[tp_na2]

                x_train_pos2 = train_test2[0]
                x_test_pos2 = train_test2[1]
                y_train_pos2 = train_test2[2]
                y_test_pos2 = train_test2[3]
                x_train_neg2 = train_test2[4]
                x_test_neg2 = train_test2[5]
                y_train_neg2 = train_test2[6]
                y_test_neg2 = train_test2[7]

                x_train = np.concatenate((x_train_pos1, x_train_neg1, x_train_pos2, x_train_neg2), axis=0)

                y_train = np.concatenate((y_train_pos1, y_train_neg1, y_train_pos2, y_train_neg2), axis=0)

                x_test1 = np.concatenate((x_test_pos1, x_test_neg1), axis=0)
                y_test1 = np.concatenate((y_test_pos1, y_test_neg1), axis=0)

                x_test2 = np.concatenate((x_test_pos2, x_test_neg2), axis=0)
                y_test2 = np.concatenate((y_test_pos2, y_test_neg2), axis=0)

                # # ######################################################################

                res_dir = './attack56/'

                from sklearn import metrics
                from sklearn.neural_network import MLPClassifier

                mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,
                                    max_iter=1500, early_stopping=True)

                # print(x_train,y_train[:,1])
                # print(np.shape(x_train), np.shape(y_train[:, 1]))
                mlp.fit(x_train, y_train[:, 1])

                print("Training set score: %f" % mlp.score(x_train, y_train[:, 1]))
                print("Test set score: %f" % mlp.score(x_test1, y_test1[:, 1]))

                y_score = mlp.predict(x_test1)
                proba = mlp.predict_proba(x_test1)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test1[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test1[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test1[:, 1], y_score)
                recall = recall_score(y_test1[:, 1], y_score)
                precision = precision_score(y_test1[:, 1], y_score)
                f1 = f1_score(y_test1[:, 1], y_score)
                auc = roc_auc_score(y_test1[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test1[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test1[i][1], prob, y_test1[i][0]]
                    tsts.append(tst)
                na = tp1 + '-' + tp2 + '-' + str(j) + '-' + str(r)
                name = ['y_score', 'y_test_grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-mlp-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])

                y_score = mlp.predict(x_test2)
                proba = mlp.predict_proba(x_test2)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test2[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test2[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test2[:, 1], y_score)
                recall = recall_score(y_test2[:, 1], y_score)
                precision = precision_score(y_test2[:, 1], y_score)
                f1 = f1_score(y_test2[:, 1], y_score)
                auc = roc_auc_score(y_test2[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test2[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test2[i][1], prob, y_test2[i][0]]
                    tsts.append(tst)
                na = tp2 + '-' + tp1 + '-' + str(j) + '-' + str(r)
                name = ['y_score', 'y_test_grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-mlp-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])

                # # ######################################################################
                # # # ######################################################################

                from sklearn.linear_model import LogisticRegression

                lr = LogisticRegression(random_state=0)
                lr.fit(x_train, y_train[:, 1])

                print("Training set score: %f" % lr.score(x_train, y_train[:, 1]))
                print("Test set score: %f" % lr.score(x_test1, y_test1[:, 1]))

                y_score = lr.predict(x_test1)
                proba = lr.predict_proba(x_test1)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test1[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test1[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test1[:, 1], y_score)
                recall = recall_score(y_test1[:, 1], y_score)
                precision = precision_score(y_test1[:, 1], y_score)
                f1 = f1_score(y_test1[:, 1], y_score)
                auc = roc_auc_score(y_test1[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test1[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test1[i][1], prob, y_test1[i][0]]
                    tsts.append(tst)
                na = tp1 + '-' + tp2 + '-' + str(j) + '-' + str(r)
                name = ['pred_label', 'grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-lr-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])

                y_score = lr.predict(x_test2)
                proba = lr.predict_proba(x_test2)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test2[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test2[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test2[:, 1], y_score)
                recall = recall_score(y_test2[:, 1], y_score)
                precision = precision_score(y_test2[:, 1], y_score)
                f1 = f1_score(y_test2[:, 1], y_score)
                auc = roc_auc_score(y_test2[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test2[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test2[i][1], prob, y_test2[i][0]]
                    tsts.append(tst)
                na = tp2 + '-' + tp1 + '-' + str(j) + '-' + str(r)
                name = ['pred_label', 'grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-lr-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])

                # # ######################################################################
                # # # ######################################################################

                from sklearn.ensemble import RandomForestClassifier

                rf = RandomForestClassifier(max_depth=150, random_state=0)
                rf.fit(x_train, y_train[:, 1])

                print("Training set score: %f" % rf.score(x_train, y_train[:, 1]))
                print("Test set score: %f" % rf.score(x_test1, y_test1[:, 1]))

                y_score = rf.predict(x_test1)
                proba = rf.predict_proba(x_test1)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test1[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test1[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test1[:, 1], y_score)
                recall = recall_score(y_test1[:, 1], y_score)
                precision = precision_score(y_test1[:, 1], y_score)
                f1 = f1_score(y_test1[:, 1], y_score)
                auc = roc_auc_score(y_test1[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test1[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test1[i][1], prob, y_test1[i][0]]
                    tsts.append(tst)
                na = tp1 + '-' + tp2 + '-' + str(j) + '-' + str(r)
                name = ['pred_label', 'grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-rf-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])

                y_score = rf.predict(x_test2)
                proba = rf.predict_proba(x_test2)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test2[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test2[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test2[:, 1], y_score)
                recall = recall_score(y_test2[:, 1], y_score)
                precision = precision_score(y_test2[:, 1], y_score)
                f1 = f1_score(y_test2[:, 1], y_score)
                auc = roc_auc_score(y_test2[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test2[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test2[i][1], prob, y_test2[i][0]]
                    tsts.append(tst)
                na = tp2 + '-' + tp1 + '-' + str(j) + '-' + str(r)
                name = ['pred_label', 'grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-rf-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])

                tp2 = tps[iii]
                tp1 = tps[ii]
                tp_na1 = str(j) + tp1
                train_test1 = np.array(train_tests[tp_na1])

                index_all = list(range(400))
                num_sample = int(r * 400)

                index = np.array(random.sample(index_all, num_sample))
                x_train_pos1 = train_test1[0][index]
                x_test_pos1 = train_test1[1]
                y_train_pos1 = train_test1[2][index]
                y_test_pos1 = train_test1[3]
                x_train_neg1 = train_test1[4][index]
                x_test_neg1 = train_test1[5]
                y_train_neg1 = train_test1[6][index]
                y_test_neg1 = train_test1[7]

                tp_na2 = str(j) + tp2
                train_test2 = train_tests[tp_na2]

                x_train_pos2 = train_test2[0]
                x_test_pos2 = train_test2[1]
                y_train_pos2 = train_test2[2]
                y_test_pos2 = train_test2[3]
                x_train_neg2 = train_test2[4]
                x_test_neg2 = train_test2[5]
                y_train_neg2 = train_test2[6]
                y_test_neg2 = train_test2[7]

                x_train = np.concatenate((x_train_pos1, x_train_neg1, x_train_pos2, x_train_neg2), axis=0)

                y_train = np.concatenate((y_train_pos1, y_train_neg1, y_train_pos2, y_train_neg2), axis=0)

                x_test1 = np.concatenate((x_test_pos1, x_test_neg1), axis=0)
                y_test1 = np.concatenate((y_test_pos1, y_test_neg1), axis=0)

                x_test2 = np.concatenate((x_test_pos2, x_test_neg2), axis=0)
                y_test2 = np.concatenate((y_test_pos2, y_test_neg2), axis=0)

                # # ######################################################################

                res_dir = './attack56/'

                from sklearn import metrics
                from sklearn.neural_network import MLPClassifier

                mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,
                                    max_iter=1500, early_stopping=True)

                # print(x_train,y_train[:,1])
                # print(np.shape(x_train), np.shape(y_train[:, 1]))
                mlp.fit(x_train, y_train[:, 1])

                print("Training set score: %f" % mlp.score(x_train, y_train[:, 1]))
                print("Test set score: %f" % mlp.score(x_test1, y_test1[:, 1]))

                y_score = mlp.predict(x_test1)
                proba = mlp.predict_proba(x_test1)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test1[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test1[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test1[:, 1], y_score)
                recall = recall_score(y_test1[:, 1], y_score)
                precision = precision_score(y_test1[:, 1], y_score)
                f1 = f1_score(y_test1[:, 1], y_score)
                auc = roc_auc_score(y_test1[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test1[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test1[i][1], prob, y_test1[i][0]]
                    tsts.append(tst)
                na = tp1 + '-' + tp2 + '-' + str(j) + '-' + str(r)
                name = ['y_score', 'y_test_grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-mlp-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])

                y_score = mlp.predict(x_test2)
                proba = mlp.predict_proba(x_test2)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test2[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test2[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test2[:, 1], y_score)
                recall = recall_score(y_test2[:, 1], y_score)
                precision = precision_score(y_test2[:, 1], y_score)
                f1 = f1_score(y_test2[:, 1], y_score)
                auc = roc_auc_score(y_test2[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test2[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test2[i][1], prob, y_test2[i][0]]
                    tsts.append(tst)
                na = tp2 + '-' + tp1 + '-' + str(j) + '-' + str(r)
                name = ['y_score', 'y_test_grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-mlp-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])

                # # ######################################################################
                # # # ######################################################################

                from sklearn.linear_model import LogisticRegression

                lr = LogisticRegression(random_state=0)
                lr.fit(x_train, y_train[:, 1])

                print("Training set score: %f" % lr.score(x_train, y_train[:, 1]))
                print("Test set score: %f" % lr.score(x_test1, y_test1[:, 1]))

                y_score = lr.predict(x_test1)
                proba = lr.predict_proba(x_test1)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test1[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test1[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test1[:, 1], y_score)
                recall = recall_score(y_test1[:, 1], y_score)
                precision = precision_score(y_test1[:, 1], y_score)
                f1 = f1_score(y_test1[:, 1], y_score)
                auc = roc_auc_score(y_test1[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test1[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test1[i][1], prob, y_test1[i][0]]
                    tsts.append(tst)
                na = tp1 + '-' + tp2 + '-' + str(j) + '-' + str(r)
                name = ['pred_label', 'grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-lr-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])

                y_score = lr.predict(x_test2)
                proba = lr.predict_proba(x_test2)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test2[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test2[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test2[:, 1], y_score)
                recall = recall_score(y_test2[:, 1], y_score)
                precision = precision_score(y_test2[:, 1], y_score)
                f1 = f1_score(y_test2[:, 1], y_score)
                auc = roc_auc_score(y_test2[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test2[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test2[i][1], prob, y_test2[i][0]]
                    tsts.append(tst)
                na = tp2 + '-' + tp1 + '-' + str(j) + '-' + str(r)
                name = ['pred_label', 'grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-lr-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])

                # # ######################################################################
                # # # ######################################################################

                from sklearn.ensemble import RandomForestClassifier

                rf = RandomForestClassifier(max_depth=150, random_state=0)
                rf.fit(x_train, y_train[:, 1])

                print("Training set score: %f" % rf.score(x_train, y_train[:, 1]))
                print("Test set score: %f" % rf.score(x_test1, y_test1[:, 1]))

                y_score = rf.predict(x_test1)
                proba = rf.predict_proba(x_test1)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test1[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test1[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test1[:, 1], y_score)
                recall = recall_score(y_test1[:, 1], y_score)
                precision = precision_score(y_test1[:, 1], y_score)
                f1 = f1_score(y_test1[:, 1], y_score)
                auc = roc_auc_score(y_test1[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test1[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test1[i][1], prob, y_test1[i][0]]
                    tsts.append(tst)
                na = tp1 + '-' + tp2 + '-' + str(j) + '-' + str(r)
                name = ['pred_label', 'grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-rf-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])

                y_score = rf.predict(x_test2)
                proba = rf.predict_proba(x_test2)
                proba = np.amax(proba, axis=1)
                print(metrics.f1_score(y_test2[:, 1], y_score, average='micro'))
                print(metrics.classification_report(y_test2[:, 1], y_score, labels=range(3)))

                acc = accuracy_score(y_test2[:, 1], y_score)
                recall = recall_score(y_test2[:, 1], y_score)
                precision = precision_score(y_test2[:, 1], y_score)
                f1 = f1_score(y_test2[:, 1], y_score)
                auc = roc_auc_score(y_test2[:, 1], proba)

                tsts = []
                for i in range(len(y_score)):
                    id_ = y_test2[i][0]
                    prob = proba[i]

                    tst = [y_score[i], y_test2[i][1], prob, y_test2[i][0]]
                    tsts.append(tst)
                na = tp2 + '-' + tp1 + '-' + str(j) + '-' + str(r)
                name = ['pred_label', 'grd', 'prob', 'index']
                result = pd.DataFrame(columns=name, data=tsts)
                result.to_csv("{}/{}-rf-12-13-embed12.csv".format(res_dir, na))
                print(acc, recall, precision, f1, auc)

                results.append([na, acc, recall, precision, f1, auc])

print(results)

name = ['name', 'acc', 'recall', 'precision', 'f1', 'auc']
result = pd.DataFrame(columns=name, data=results)
result.to_csv("{}/results_attack56.csv".format(res_dir))