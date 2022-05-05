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
    # embeddings1=np.mean(embeddings,axis=0)
    # embeddings2 = np.mean(embeddings, axis=1)
    # embeddings3 = np.amax(embeddings, axis=0)
    embeddings4 = np.amax(embeddings, axis=1).  #max-pooling
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
    # print('dataMat:', dataMat)
    # print(np.shape(dataMat))
    #print np.array(dataMat)
    #embeddings = np.array(dataMat)
    posts=dataMat
    return (posts)

# s=0
tps=['pokec','fb','pubmed']
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
                feat_embed2.append(np.array(embed2).flatten())

                feat_embed12.append(np.concatenate((np.array(embed1).flatten(), (np.array(embed2).flatten())), axis=0))
                feat_embed1.append(np.array(embed1).flatten())
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
k = [0,1,2,3]
tps=['pokec','fb','pubmed']
for tp1 in tps:
    for j in k:
        ft_=feats[tp1][j]
        ft_neg = feats_neg[tp1][j]

        na=tp1+'-'+ft_name[j]

        x_train_pos, x_test_pos, y_train_pos, y_test_pos  = train_test_split(ft_, label,test_size=0.3, random_state=42)
        x_train_neg, x_test_neg, y_train_neg, y_test_neg = train_test_split(ft_neg, label_neg, test_size=0.3, random_state=42)

        x_train_pos1, x_test_pos1, y_train_pos1, y_test_pos1  = train_test_split(x_train_pos, y_train_pos,test_size=0.2, random_state=42)
        x_train_neg1, x_test_neg1, y_train_neg1, y_test_neg1 = train_test_split(x_train_neg, y_train_neg, test_size=0.2, random_state=42)

        x_train_pos2, x_test_pos2, y_train_pos2, y_test_pos2 = train_test_split(x_train_pos1, y_train_pos1,
                                                                                test_size=0.25, random_state=42)
        x_train_neg2, x_test_neg2, y_train_neg2, y_test_neg2 = train_test_split(x_train_neg1, y_train_neg1,
                                                                                test_size=0.25, random_state=42)

        x_train_pos3, x_test_pos3, y_train_pos3, y_test_pos3 = train_test_split(x_train_pos2, y_train_pos2,
                                                                                test_size=1/3, random_state=42)
        x_train_neg3, x_test_neg3, y_train_neg3, y_test_neg3 = train_test_split(x_train_neg2, y_train_neg2,
                                                                                test_size=1/3, random_state=42)

        x_train_pos4, x_test_pos4, y_train_pos4, y_test_pos4 = train_test_split(x_train_pos3, y_train_pos3,
                                                                                test_size=0.5, random_state=42)
        x_train_neg4, x_test_neg4, y_train_neg4, y_test_neg4 = train_test_split(x_train_neg3, y_train_neg3,
                                                                                test_size=0.5, random_state=42)

        x_test_pos5= x_train_pos4
        y_test_pos5 =y_train_pos4
        x_test_neg5 =x_train_neg4
        y_test_neg5 =y_train_neg4

        x_train1 = np.concatenate((x_test_pos1, x_test_neg1), axis=0)
        y_train1 = np.concatenate((y_test_pos1,y_test_neg1), axis=0)

        x_train2 = np.concatenate((x_test_pos2, x_test_neg2), axis=0)
        y_train2 = np.concatenate((y_test_pos2, y_test_neg2), axis=0)

        x_train3 = np.concatenate((x_test_pos3, x_test_neg3), axis=0)
        y_train3 = np.concatenate((y_test_pos3, y_test_neg3), axis=0)

        x_train4 = np.concatenate((x_test_pos4, x_test_neg4), axis=0)
        y_train4 = np.concatenate((y_test_pos4, y_test_neg4), axis=0)

        x_train5 = np.concatenate((x_test_pos5, x_test_neg5), axis=0)
        y_train5 = np.concatenate((y_test_pos5, y_test_neg5), axis=0)


        x_test = np.concatenate((x_test_pos, x_test_neg), axis=0)
        y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)


        # X_train, X_test, y_train, y_test = train_test_split(sim_matrix, y_label, test_size=0.3, random_state=42)
        #
        # # ######################################################################

        res_dir='./ensemble-baseline/'

        from sklearn import metrics
        from sklearn.neural_network import MLPClassifier

        mlp1 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,max_iter=1500,early_stopping=True)

        # print(x_train,y_train[:,1])
        # print(np.shape(x_train), np.shape(y_train[:, 1]))
        mlp1.fit(x_train1, y_train1[:,1])

        mlp2 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1, max_iter=1500,
                            early_stopping=True)

        mlp2.fit(x_train2, y_train2[:, 1])

        mlp3 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1, max_iter=1500,
                            early_stopping=True)

        mlp3.fit(x_train3, y_train3[:, 1])

        mlp4 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1, max_iter=1500,
                            early_stopping=True)

        mlp4.fit(x_train4, y_train4[:, 1])

        mlp5 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1, max_iter=1500,
                            early_stopping=True)

        mlp5.fit(x_train5, y_train5[:, 1])

        y_score1 = mlp1.predict(x_test)
        proba1 = mlp1.predict_proba(x_test)
        proba1=np.amax(proba1,axis=1)

        y_score2 = mlp2.predict(x_test)
        proba2 = mlp2.predict_proba(x_test)
        proba2 = np.amax(proba2, axis=1)

        y_score3 = mlp3.predict(x_test)
        proba3 = mlp3.predict_proba(x_test)
        proba3 = np.amax(proba3, axis=1)


        y_score4 = mlp4.predict(x_test)
        proba4 = mlp4.predict_proba(x_test)
        proba4 = np.amax(proba4, axis=1)


        y_score5 = mlp5.predict(x_test)
        proba5 = mlp5.predict_proba(x_test)
        proba5 = np.amax(proba5, axis=1)

        y_score = []

        y_score.append(y_score1)
        y_score.append(y_score2)
        y_score.append(y_score3)
        y_score.append(y_score4)
        y_score.append(y_score5)

        y_score = np.array(y_score)

        y_score = np.sum(y_score, axis=0)

        lb=[]

        for sc in y_score:
            if sc>=3:
                lb.append(1)
            else:
                lb.append(0)

        lb=np.array(lb)

        acc = accuracy_score(y_test[:,1],lb)
        recall = recall_score(y_test[:,1],lb)
        precision = precision_score(y_test[:,1],lb)
        f1 = f1_score(y_test[:,1],lb)
        # auc = roc_auc_score(y_test[:,1], proba)


        tsts = []
        for i in range(len(y_score)):
            id_=y_test[i][0]

            tst = [y_score[i], y_test[i][1], y_test[i][0]]
            tsts.append(tst)
        name = ['y_score', 'y_test_grd', 'index']
        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}/{}-mlp-12-13.csv".format(res_dir,na))
        print(acc,recall,precision,f1)

        results.append([na,acc,recall,precision,f1])

        # # ######################################################################
        # # # ######################################################################

        from sklearn.linear_model import LogisticRegression

        lr1 = LogisticRegression(random_state=0)

        # print(x_train,y_train[:,1])
        # print(np.shape(x_train), np.shape(y_train[:, 1]))
        lr1.fit(x_train1, y_train1[:, 1])

        lr2 = LogisticRegression(random_state=0)
        lr2.fit(x_train2, y_train2[:, 1])

        lr3 = LogisticRegression(random_state=0)
        lr3.fit(x_train3, y_train3[:, 1])

        lr4 = LogisticRegression(random_state=0)
        lr4.fit(x_train4, y_train4[:, 1])

        lr5 = LogisticRegression(random_state=0)
        lr5.fit(x_train5, y_train5[:, 1])

        y_score1 = lr1.predict(x_test)
        proba1 = lr1.predict_proba(x_test)
        proba1 = np.amax(proba1, axis=1)

        y_score2 = lr2.predict(x_test)
        proba2 = lr2.predict_proba(x_test)
        proba2 = np.amax(proba2, axis=1)

        y_score3 = lr3.predict(x_test)
        proba3 = lr3.predict_proba(x_test)
        proba3 = np.amax(proba3, axis=1)

        y_score4 = lr4.predict(x_test)
        proba4 = lr4.predict_proba(x_test)
        proba4 = np.amax(proba4, axis=1)

        y_score5 = lr5.predict(x_test)
        proba5 = lr5.predict_proba(x_test)
        proba5 = np.amax(proba5, axis=1)

        y_score = []

        y_score.append(y_score1)
        y_score.append(y_score2)
        y_score.append(y_score3)
        y_score.append(y_score4)
        y_score.append(y_score5)

        y_score = np.array(y_score)

        y_score = np.sum(y_score, axis=0)
        lb = []

        for sc in y_score:
            if sc >= 3:
                lb.append(1)
            else:
                lb.append(0)

        lb = np.array(lb)

        acc = accuracy_score(y_test[:, 1], lb)
        recall = recall_score(y_test[:, 1], lb)
        precision = precision_score(y_test[:, 1], lb)
        f1 = f1_score(y_test[:, 1], lb)
        # auc = roc_auc_score(y_test[:,1], proba)


        tsts = []
        for i in range(len(y_score)):
            id_ = y_test[i][0]

            tst = [y_score[i], y_test[i][1], y_test[i][0]]
            tsts.append(tst)
        name = ['y_score', 'y_test_grd', 'index']
        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}/{}-lr-12-13.csv".format(res_dir, na))
        print(acc, recall, precision, f1)

        results.append([na, acc, recall, precision, f1])




        # # ######################################################################
        # # # ######################################################################

        from sklearn.ensemble import RandomForestClassifier

        rf1 = RandomForestClassifier(max_depth=150, random_state=0)
        # print(x_train,y_train[:,1])
        # print(np.shape(x_train), np.shape(y_train[:, 1]))
        rf1.fit(x_train1, y_train1[:, 1])

        rf2 = RandomForestClassifier(max_depth=150, random_state=0)
        rf2.fit(x_train2, y_train2[:, 1])

        rf3 = RandomForestClassifier(max_depth=150, random_state=0)
        rf3.fit(x_train3, y_train3[:, 1])

        rf4 = RandomForestClassifier(max_depth=150, random_state=0)
        rf4.fit(x_train4, y_train4[:, 1])

        rf5 = RandomForestClassifier(max_depth=150, random_state=0)
        rf5.fit(x_train5, y_train5[:, 1])

        y_score1 = rf1.predict(x_test)
        proba1 = rf1.predict_proba(x_test)
        proba1 = np.amax(proba1, axis=1)

        y_score2 = rf2.predict(x_test)
        proba2 = rf2.predict_proba(x_test)
        proba2 = np.amax(proba2, axis=1)

        y_score3 = rf3.predict(x_test)
        proba3 = rf3.predict_proba(x_test)
        proba3 = np.amax(proba3, axis=1)

        y_score4 = rf4.predict(x_test)
        proba4 = rf4.predict_proba(x_test)
        proba4 = np.amax(proba4, axis=1)

        y_score5 = rf5.predict(x_test)
        proba5 = rf5.predict_proba(x_test)
        proba5 = np.amax(proba5, axis=1)

        y_score = []

        y_score.append(y_score1)
        y_score.append(y_score2)
        y_score.append(y_score3)
        y_score.append(y_score4)
        y_score.append(y_score5)

        y_score = np.array(y_score)

        y_score = np.sum(y_score, axis=0)

        lb = []

        for sc in y_score:
            if sc >= 3:
                lb.append(1)
            else:
                lb.append(0)

        lb = np.array(lb)

        acc = accuracy_score(y_test[:, 1], lb)
        recall = recall_score(y_test[:, 1], lb)
        precision = precision_score(y_test[:, 1], lb)
        f1 = f1_score(y_test[:, 1], lb)
        # auc = roc_auc_score(y_test[:,1], proba)


        tsts = []
        for i in range(len(y_score)):
            id_ = y_test[i][0]

            tst = [y_score[i], y_test[i][1], y_test[i][0]]
            tsts.append(tst)
        name = ['y_score', 'y_test_grd', 'index']
        result = pd.DataFrame(columns=name, data=tsts)
        result.to_csv("{}/{}-rf-12-13.csv".format(res_dir, na))
        print(acc, recall, precision, f1)

        results.append([na, acc, recall, precision, f1])

print(results)
name = ['name','acc', 'recall', 'precision', 'f1']
result = pd.DataFrame(columns=name, data=results)
result.to_csv("{}/results_baseline-5classiers.csv".format(res_dir))
