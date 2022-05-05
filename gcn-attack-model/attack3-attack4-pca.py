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
    # print('dataMat:', dataMat)
    # print(np.shape(dataMat))
    #print np.array(dataMat)
    #embeddings = np.array(dataMat)
    embeddings=dataMat
    return (embeddings)

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
fts=['embed1','embed2','post']
feats={}
feats_neg={}
i=0
for tp1 in tps:

    res_dir = './pca-tranfer/'
    tp = 'pca'
    para_list = []
    para_list_neg = []

    for ft in fts:
        label = []
        label_neg = []
        para = []
        para_neg = []

        ff = "./{}/{}-99-5-{}-0.99.pkl".format(tp, ft,tp1)


        with open(ff, "rb") as f:
            paras = pk.load(f, encoding='latin1')


        for i in range(100):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(100,200):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(200,300):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(300,400):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(400,500):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(500,600):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(600,700):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(700,800):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(800,900):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(900,1000):
            label.append([i, 1])
            para.append(paras[i])

        para_list.append(np.array(para))
        para_list_neg.append(np.array(para_neg))


    feats[tp1] = para_list
    feats_neg[tp1] = para_list_neg

ft_name=['feat_embed1','feat_embed2','feat_post']
from sklearn.model_selection import train_test_split

results=[]
# k=[1,4,6]
k = [0,1,2]
# tps1=['edu-gender-2']
tps=['gender','edu-gender','pubmed-small-3.16']
for tp1 in tps:
    for tp2 in tps:
        # if tp1==tp2:
        #     continue
        for j in k:
            ft_=feats[tp1][j]
            ft_neg = feats_neg[tp1][j]
            ft2_=feats[tp2][j]
            ft2_neg = feats_neg[tp2][j]
            na=tp1+'-'+tp2+'-'+ft_name[j]

            if np.shape(ft_)[1]!=np.shape(ft2_)[1]:
                continue

            # print(np.shape(ft_))
            # print(np.shape(ft_neg))

            x_train_pos, x_test_pos, y_train_pos, y_test_pos  = train_test_split(ft_, label,test_size=0.3, random_state=42)
            x_train_neg, x_test_neg, y_train_neg, y_test_neg = train_test_split(ft_neg, label_neg, test_size=0.3, random_state=42)

            x_train = np.concatenate((x_train_pos, x_train_neg), axis=0)
            x_test = np.concatenate((x_test_pos, x_test_neg), axis=0)
            y_train = np.concatenate((y_train_pos,y_train_neg), axis=0)
            y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)

            x_train_pos2, x_test_pos2, y_train_pos2, y_test_pos2 = train_test_split(ft2_, label, test_size=0.3,
                                                                                random_state=42)
            x_train_neg2, x_test_neg2, y_train_neg2, y_test_neg2 = train_test_split(ft2_neg, label_neg, test_size=0.3,
                                                                                random_state=42)

            x_train2 = np.concatenate((x_train_pos2, x_train_neg2), axis=0)
            x_test2 = np.concatenate((x_test_pos2, x_test_neg2), axis=0)
            y_train2 = np.concatenate((y_train_pos2, y_train_neg2), axis=0)
            y_test2 = np.concatenate((y_test_pos2, y_test_neg2), axis=0)

            # X_train, X_test, y_train, y_test = train_test_split(sim_matrix, y_label, test_size=0.3, random_state=42)
            #
            # # ######################################################################

            res_dir='./pca-transfer/'

            from sklearn import metrics
            from sklearn.neural_network import MLPClassifier

            mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,max_iter=1500,early_stopping=True)

            # print(x_train,y_train[:,1])
            # print(np.shape(x_train), np.shape(y_train[:, 1]))
            mlp.fit(x_train, y_train[:,1])

            print("Training set score: %f" % mlp.score(x_train, y_train[:,1]))
            print("Test set score: %f" % mlp.score(x_test2, y_test2[:,1]))

            y_score = mlp.predict(x_test2)
            proba = mlp.predict_proba(x_test2)
            proba=np.amax(proba,axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))
            # report = metrics.classification_report(y_test2[:,1], y_score, labels=range(3), output_dict=True)
            #
            # out = "{}/{}-mlp-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test2[i][0]
                prob = proba[i]

                tst = [y_score[i], y_test2[i][1],prob, y_test2[i][0]]
                tsts.append(tst)
            name = ['y_score', 'y_test_grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-mlp-12-13-0.99.csv".format(res_dir,na))
            print(acc,recall,precision,f1,auc)

            results.append([na,acc,recall,precision,f1,auc])

            # # ######################################################################
            # # # ######################################################################

            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(random_state=0)
            lr.fit(x_train, y_train[:,1])


            print("Training set score: %f" % lr.score(x_train, y_train[:,1]))
            print("Test set score: %f" % lr.score(x_test2, y_test2[:,1]))

            y_score = lr.predict(x_test2)
            proba = lr.predict_proba(x_test2)
            proba = np.amax(proba, axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))

            # report=metrics.classification_report(y_test2[:,1], y_score, labels=range(3),output_dict=True)
            # # print(report)
            #
            # out = "{}/{}-lr-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test[i][0]
                prob=proba[i]

                tst = [y_score[i], y_test2[i][1], prob,y_test2[i][0]]
                tsts.append(tst)
            name = ['pred_label', 'grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-lr-12-13-0.99.csv".format(res_dir,na))
            print(acc, recall, precision, f1, auc)

            results.append([na,acc, recall, precision, f1, auc])

            # # ######################################################################
            # # # ######################################################################

            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(max_depth=150, random_state=0)
            rf.fit(x_train, y_train[:,1])

            print("Training set score: %f" % rf.score(x_train, y_train[:,1]))
            print("Test set score: %f" % rf.score(x_test2, y_test2[:,1]))

            y_score = rf.predict(x_test2)
            proba = rf.predict_proba(x_test2)
            proba = np.amax(proba, axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))

            # report=metrics.classification_report(y_test2[:,1], y_score, labels=range(3),output_dict=True)
            # # print(report)
            #
            # out = "{}/{}-rf-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test2[i][0]
                prob=proba[i]

                tst = [y_score[i], y_test2[i][1], prob,y_test2[i][0]]
                tsts.append(tst)
            name = ['pred_label', 'grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-rf-12-13-0.99.csv".format(res_dir,na))
            print(acc, recall, precision, f1, auc)

            results.append([na,acc, recall, precision, f1, auc])

print(results)
name = ['name','acc', 'recall', 'precision', 'f1', 'auc']
result = pd.DataFrame(columns=name, data=results)
result.to_csv("{}/results_transfer-0.99.csv".format(res_dir))



dataset=['pokec','fb','pubmed']
tps=['pokec-interintra','fb-interintra','pubmed-interintra']
fts=['embed1','embed2','post']
feats={}
feats_neg={}
i=0
for tp1 in tps:

    res_dir = './pca-tranfer/'
    tp = 'pca'
    para_list = []
    para_list_neg = []

    for ft in fts:
        label = []
        label_neg = []
        para = []
        para_neg = []

        ff = "./{}/{}-99-5-{}-0.99.pkl".format(tp, ft,tp1)


        with open(ff, "rb") as f:
            paras = pk.load(f, encoding='latin1')


        for i in range(100):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(100,200):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(200,300):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(300,400):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(400,500):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(500,600):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(600,700):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(700,800):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(800,900):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(900,1000):
            label.append([i, 1])
            para.append(paras[i])

        para_list.append(np.array(para))
        para_list_neg.append(np.array(para_neg))


    feats[tp1] = para_list
    feats_neg[tp1] = para_list_neg

ft_name=['feat_embed1','feat_embed2','feat_post']
from sklearn.model_selection import train_test_split

results=[]
# k=[1,4,6]
k = [0,1,2]
# tps1=['edu-gender-2']
tps=['pokec-interintra','fb-interintra','pubmed-interintra']
for tp1 in tps:
    for tp2 in tps:
        # if tp1==tp2:
        #     continue
        for j in k:
            ft_=feats[tp1][j]
            ft_neg = feats_neg[tp1][j]
            ft2_=feats[tp2][j]
            ft2_neg = feats_neg[tp2][j]
            na=tp1+'-'+tp2+'-'+ft_name[j]

            if np.shape(ft_)[1]!=np.shape(ft2_)[1]:
                continue

            # print(np.shape(ft_))
            # print(np.shape(ft_neg))

            x_train_pos, x_test_pos, y_train_pos, y_test_pos  = train_test_split(ft_, label,test_size=0.3, random_state=42)
            x_train_neg, x_test_neg, y_train_neg, y_test_neg = train_test_split(ft_neg, label_neg, test_size=0.3, random_state=42)

            x_train = np.concatenate((x_train_pos, x_train_neg), axis=0)
            x_test = np.concatenate((x_test_pos, x_test_neg), axis=0)
            y_train = np.concatenate((y_train_pos,y_train_neg), axis=0)
            y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)

            x_train_pos2, x_test_pos2, y_train_pos2, y_test_pos2 = train_test_split(ft2_, label, test_size=0.3,
                                                                                random_state=42)
            x_train_neg2, x_test_neg2, y_train_neg2, y_test_neg2 = train_test_split(ft2_neg, label_neg, test_size=0.3,
                                                                                random_state=42)

            x_train2 = np.concatenate((x_train_pos2, x_train_neg2), axis=0)
            x_test2 = np.concatenate((x_test_pos2, x_test_neg2), axis=0)
            y_train2 = np.concatenate((y_train_pos2, y_train_neg2), axis=0)
            y_test2 = np.concatenate((y_test_pos2, y_test_neg2), axis=0)

            # X_train, X_test, y_train, y_test = train_test_split(sim_matrix, y_label, test_size=0.3, random_state=42)
            #
            # # ######################################################################

            res_dir='./pca-transfer/'

            from sklearn import metrics
            from sklearn.neural_network import MLPClassifier

            mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,max_iter=1500,early_stopping=True)

            # print(x_train,y_train[:,1])
            # print(np.shape(x_train), np.shape(y_train[:, 1]))
            mlp.fit(x_train, y_train[:,1])

            print("Training set score: %f" % mlp.score(x_train, y_train[:,1]))
            print("Test set score: %f" % mlp.score(x_test2, y_test2[:,1]))

            y_score = mlp.predict(x_test2)
            proba = mlp.predict_proba(x_test2)
            proba=np.amax(proba,axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))
            # report = metrics.classification_report(y_test2[:,1], y_score, labels=range(3), output_dict=True)
            #
            # out = "{}/{}-mlp-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test2[i][0]
                prob = proba[i]

                tst = [y_score[i], y_test2[i][1],prob, y_test2[i][0]]
                tsts.append(tst)
            name = ['y_score', 'y_test_grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-mlp-12-13-0.99.csv".format(res_dir,na))
            print(acc,recall,precision,f1,auc)

            results.append([na,acc,recall,precision,f1,auc])

            # # ######################################################################
            # # # ######################################################################

            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(random_state=0)
            lr.fit(x_train, y_train[:,1])


            print("Training set score: %f" % lr.score(x_train, y_train[:,1]))
            print("Test set score: %f" % lr.score(x_test2, y_test2[:,1]))

            y_score = lr.predict(x_test2)
            proba = lr.predict_proba(x_test2)
            proba = np.amax(proba, axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))

            # report=metrics.classification_report(y_test2[:,1], y_score, labels=range(3),output_dict=True)
            # # print(report)
            #
            # out = "{}/{}-lr-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test[i][0]
                prob=proba[i]

                tst = [y_score[i], y_test2[i][1], prob,y_test2[i][0]]
                tsts.append(tst)
            name = ['pred_label', 'grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-lr-12-13-0.99.csv".format(res_dir,na))
            print(acc, recall, precision, f1, auc)

            results.append([na,acc, recall, precision, f1, auc])

            # # ######################################################################
            # # # ######################################################################

            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(max_depth=150, random_state=0)
            rf.fit(x_train, y_train[:,1])

            print("Training set score: %f" % rf.score(x_train, y_train[:,1]))
            print("Test set score: %f" % rf.score(x_test2, y_test2[:,1]))

            y_score = rf.predict(x_test2)
            proba = rf.predict_proba(x_test2)
            proba = np.amax(proba, axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))

            # report=metrics.classification_report(y_test2[:,1], y_score, labels=range(3),output_dict=True)
            # # print(report)
            #
            # out = "{}/{}-rf-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test2[i][0]
                prob=proba[i]

                tst = [y_score[i], y_test2[i][1], prob,y_test2[i][0]]
                tsts.append(tst)
            name = ['pred_label', 'grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-rf-12-13-0.99.csv".format(res_dir,na))
            print(acc, recall, precision, f1, auc)

            results.append([na,acc, recall, precision, f1, auc])

print(results)
name = ['name','acc', 'recall', 'precision', 'f1', 'auc']
result = pd.DataFrame(columns=name, data=results)
result.to_csv("{}/results_transfer-interintra-0.99.csv".format(res_dir))


# s=0
dataset=['pokec','fb','pubmed']
tps=['pokec','fb','pubmed']
fts=['embed1','embed2','post']
feats={}
feats_neg={}
i=0
for tp1 in tps:

    res_dir = './pca-tranfer/'
    tp = 'pca'
    para_list = []
    para_list_neg = []

    for ft in fts:
        label = []
        label_neg = []
        para = []
        para_neg = []

        ff = "./{}/{}-99-5-{}-0.95.pkl".format(tp, ft,tp1)


        with open(ff, "rb") as f:
            paras = pk.load(f, encoding='latin1')


        for i in range(100):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(100,200):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(200,300):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(300,400):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(400,500):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(500,600):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(600,700):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(700,800):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(800,900):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(900,1000):
            label.append([i, 1])
            para.append(paras[i])

        para_list.append(np.array(para))
        para_list_neg.append(np.array(para_neg))


    feats[tp1] = para_list
    feats_neg[tp1] = para_list_neg

ft_name=['feat_embed1','feat_embed2','feat_post']
from sklearn.model_selection import train_test_split

results=[]
# k=[1,4,6]
k = [0,1,2]
# tps1=['edu-gender-2']
tps=['pokec','fb','pubmed']
for tp1 in tps:
    for tp2 in tps:
        # if tp1==tp2:
        #     continue
        for j in k:
            ft_=feats[tp1][j]
            ft_neg = feats_neg[tp1][j]
            ft2_=feats[tp2][j]
            ft2_neg = feats_neg[tp2][j]
            na=tp1+'-'+tp2+'-'+ft_name[j]

            if np.shape(ft_)[1]!=np.shape(ft2_)[1]:
                continue

            # print(np.shape(ft_))
            # print(np.shape(ft_neg))

            x_train_pos, x_test_pos, y_train_pos, y_test_pos  = train_test_split(ft_, label,test_size=0.3, random_state=42)
            x_train_neg, x_test_neg, y_train_neg, y_test_neg = train_test_split(ft_neg, label_neg, test_size=0.3, random_state=42)

            x_train = np.concatenate((x_train_pos, x_train_neg), axis=0)
            x_test = np.concatenate((x_test_pos, x_test_neg), axis=0)
            y_train = np.concatenate((y_train_pos,y_train_neg), axis=0)
            y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)

            x_train_pos2, x_test_pos2, y_train_pos2, y_test_pos2 = train_test_split(ft2_, label, test_size=0.3,
                                                                                random_state=42)
            x_train_neg2, x_test_neg2, y_train_neg2, y_test_neg2 = train_test_split(ft2_neg, label_neg, test_size=0.3,
                                                                                random_state=42)

            x_train2 = np.concatenate((x_train_pos2, x_train_neg2), axis=0)
            x_test2 = np.concatenate((x_test_pos2, x_test_neg2), axis=0)
            y_train2 = np.concatenate((y_train_pos2, y_train_neg2), axis=0)
            y_test2 = np.concatenate((y_test_pos2, y_test_neg2), axis=0)

            # X_train, X_test, y_train, y_test = train_test_split(sim_matrix, y_label, test_size=0.3, random_state=42)
            #
            # # ######################################################################

            res_dir='./pca-transfer/'

            from sklearn import metrics
            from sklearn.neural_network import MLPClassifier

            mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,max_iter=1500,early_stopping=True)

            # print(x_train,y_train[:,1])
            # print(np.shape(x_train), np.shape(y_train[:, 1]))
            mlp.fit(x_train, y_train[:,1])

            print("Training set score: %f" % mlp.score(x_train, y_train[:,1]))
            print("Test set score: %f" % mlp.score(x_test2, y_test2[:,1]))

            y_score = mlp.predict(x_test2)
            proba = mlp.predict_proba(x_test2)
            proba=np.amax(proba,axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))
            # report = metrics.classification_report(y_test2[:,1], y_score, labels=range(3), output_dict=True)
            #
            # out = "{}/{}-mlp-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test2[i][0]
                prob = proba[i]

                tst = [y_score[i], y_test2[i][1],prob, y_test2[i][0]]
                tsts.append(tst)
            name = ['y_score', 'y_test_grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-mlp-12-13-0.95.csv".format(res_dir,na))
            print(acc,recall,precision,f1,auc)

            results.append([na,acc,recall,precision,f1,auc])

            # # ######################################################################
            # # # ######################################################################

            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(random_state=0)
            lr.fit(x_train, y_train[:,1])


            print("Training set score: %f" % lr.score(x_train, y_train[:,1]))
            print("Test set score: %f" % lr.score(x_test2, y_test2[:,1]))

            y_score = lr.predict(x_test2)
            proba = lr.predict_proba(x_test2)
            proba = np.amax(proba, axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))

            # report=metrics.classification_report(y_test2[:,1], y_score, labels=range(3),output_dict=True)
            # # print(report)
            #
            # out = "{}/{}-lr-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test[i][0]
                prob=proba[i]

                tst = [y_score[i], y_test2[i][1], prob,y_test2[i][0]]
                tsts.append(tst)
            name = ['pred_label', 'grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-lr-12-13-0.95.csv".format(res_dir,na))
            print(acc, recall, precision, f1, auc)

            results.append([na,acc, recall, precision, f1, auc])

            # # ######################################################################
            # # # ######################################################################

            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(max_depth=150, random_state=0)
            rf.fit(x_train, y_train[:,1])

            print("Training set score: %f" % rf.score(x_train, y_train[:,1]))
            print("Test set score: %f" % rf.score(x_test2, y_test2[:,1]))

            y_score = rf.predict(x_test2)
            proba = rf.predict_proba(x_test2)
            proba = np.amax(proba, axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))

            # report=metrics.classification_report(y_test2[:,1], y_score, labels=range(3),output_dict=True)
            # # print(report)
            #
            # out = "{}/{}-rf-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test2[i][0]
                prob=proba[i]

                tst = [y_score[i], y_test2[i][1], prob,y_test2[i][0]]
                tsts.append(tst)
            name = ['pred_label', 'grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-rf-12-13-0.95.csv".format(res_dir,na))
            print(acc, recall, precision, f1, auc)

            results.append([na,acc, recall, precision, f1, auc])

print(results)
name = ['name','acc', 'recall', 'precision', 'f1', 'auc']
result = pd.DataFrame(columns=name, data=results)
result.to_csv("{}/results_transfer-0.95.csv".format(res_dir))



dataset=['pokec','fb','pubmed']
tps=['pokec-interintra','fb-interintra','pubmed-interintra']
fts=['embed1','embed2','post']
feats={}
feats_neg={}
i=0
for tp1 in tps:

    res_dir = './pca-tranfer/'
    tp = 'pca'
    para_list = []
    para_list_neg = []

    for ft in fts:
        label = []
        label_neg = []
        para = []
        para_neg = []

        ff = "./{}/{}-99-5-{}-0.95.pkl".format(tp, ft,tp1)


        with open(ff, "rb") as f:
            paras = pk.load(f, encoding='latin1')


        for i in range(100):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(100,200):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(200,300):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(300,400):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(400,500):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(500,600):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(600,700):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(700,800):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(800,900):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(900,1000):
            label.append([i, 1])
            para.append(paras[i])

        para_list.append(np.array(para))
        para_list_neg.append(np.array(para_neg))


    feats[tp1] = para_list
    feats_neg[tp1] = para_list_neg

ft_name=['feat_embed1','feat_embed2','feat_post']
from sklearn.model_selection import train_test_split

results=[]
# k=[1,4,6]
k = [0,1,2]
# tps1=['edu-gender-2']
tps=['pokec-interintra','fb-interintra','pubmed-interintra']
for tp1 in tps:
    for tp2 in tps:
        # if tp1==tp2:
        #     continue
        for j in k:
            ft_=feats[tp1][j]
            ft_neg = feats_neg[tp1][j]
            ft2_=feats[tp2][j]
            ft2_neg = feats_neg[tp2][j]
            na=tp1+'-'+tp2+'-'+ft_name[j]

            if np.shape(ft_)[1]!=np.shape(ft2_)[1]:
                continue

            # print(np.shape(ft_))
            # print(np.shape(ft_neg))

            x_train_pos, x_test_pos, y_train_pos, y_test_pos  = train_test_split(ft_, label,test_size=0.3, random_state=42)
            x_train_neg, x_test_neg, y_train_neg, y_test_neg = train_test_split(ft_neg, label_neg, test_size=0.3, random_state=42)

            x_train = np.concatenate((x_train_pos, x_train_neg), axis=0)
            x_test = np.concatenate((x_test_pos, x_test_neg), axis=0)
            y_train = np.concatenate((y_train_pos,y_train_neg), axis=0)
            y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)

            x_train_pos2, x_test_pos2, y_train_pos2, y_test_pos2 = train_test_split(ft2_, label, test_size=0.3,
                                                                                random_state=42)
            x_train_neg2, x_test_neg2, y_train_neg2, y_test_neg2 = train_test_split(ft2_neg, label_neg, test_size=0.3,
                                                                                random_state=42)

            x_train2 = np.concatenate((x_train_pos2, x_train_neg2), axis=0)
            x_test2 = np.concatenate((x_test_pos2, x_test_neg2), axis=0)
            y_train2 = np.concatenate((y_train_pos2, y_train_neg2), axis=0)
            y_test2 = np.concatenate((y_test_pos2, y_test_neg2), axis=0)

            # X_train, X_test, y_train, y_test = train_test_split(sim_matrix, y_label, test_size=0.3, random_state=42)
            #
            # # ######################################################################

            res_dir='./pca-transfer/'

            from sklearn import metrics
            from sklearn.neural_network import MLPClassifier

            mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,max_iter=1500,early_stopping=True)

            # print(x_train,y_train[:,1])
            # print(np.shape(x_train), np.shape(y_train[:, 1]))
            mlp.fit(x_train, y_train[:,1])

            print("Training set score: %f" % mlp.score(x_train, y_train[:,1]))
            print("Test set score: %f" % mlp.score(x_test2, y_test2[:,1]))

            y_score = mlp.predict(x_test2)
            proba = mlp.predict_proba(x_test2)
            proba=np.amax(proba,axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))
            # report = metrics.classification_report(y_test2[:,1], y_score, labels=range(3), output_dict=True)
            #
            # out = "{}/{}-mlp-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test2[i][0]
                prob = proba[i]

                tst = [y_score[i], y_test2[i][1],prob, y_test2[i][0]]
                tsts.append(tst)
            name = ['y_score', 'y_test_grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-mlp-12-13-0.95.csv".format(res_dir,na))
            print(acc,recall,precision,f1,auc)

            results.append([na,acc,recall,precision,f1,auc])

            # # ######################################################################
            # # # ######################################################################

            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(random_state=0)
            lr.fit(x_train, y_train[:,1])


            print("Training set score: %f" % lr.score(x_train, y_train[:,1]))
            print("Test set score: %f" % lr.score(x_test2, y_test2[:,1]))

            y_score = lr.predict(x_test2)
            proba = lr.predict_proba(x_test2)
            proba = np.amax(proba, axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))

            # report=metrics.classification_report(y_test2[:,1], y_score, labels=range(3),output_dict=True)
            # # print(report)
            #
            # out = "{}/{}-lr-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test[i][0]
                prob=proba[i]

                tst = [y_score[i], y_test2[i][1], prob,y_test2[i][0]]
                tsts.append(tst)
            name = ['pred_label', 'grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-lr-12-13-0.95.csv".format(res_dir,na))
            print(acc, recall, precision, f1, auc)

            results.append([na,acc, recall, precision, f1, auc])

            # # ######################################################################
            # # # ######################################################################

            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(max_depth=150, random_state=0)
            rf.fit(x_train, y_train[:,1])

            print("Training set score: %f" % rf.score(x_train, y_train[:,1]))
            print("Test set score: %f" % rf.score(x_test2, y_test2[:,1]))

            y_score = rf.predict(x_test2)
            proba = rf.predict_proba(x_test2)
            proba = np.amax(proba, axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))

            # report=metrics.classification_report(y_test2[:,1], y_score, labels=range(3),output_dict=True)
            # # print(report)
            #
            # out = "{}/{}-rf-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test2[i][0]
                prob=proba[i]

                tst = [y_score[i], y_test2[i][1], prob,y_test2[i][0]]
                tsts.append(tst)
            name = ['pred_label', 'grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-rf-12-13-0.95.csv".format(res_dir,na))
            print(acc, recall, precision, f1, auc)

            results.append([na,acc, recall, precision, f1, auc])

print(results)
name = ['name','acc', 'recall', 'precision', 'f1', 'auc']
result = pd.DataFrame(columns=name, data=results)
result.to_csv("{}/results_transfer-interintra-0.95.csv".format(res_dir))







# s=0
dataset=['pokec','fb','pubmed']
tps=['pokec','fb','pubmed']
fts=['embed1','embed2','post']
feats={}
feats_neg={}
i=0
for tp1 in tps:

    res_dir = './pca-tranfer/'
    tp = 'pca'
    para_list = []
    para_list_neg = []

    for ft in fts:
        label = []
        label_neg = []
        para = []
        para_neg = []

        ff = "./{}/{}-99-5-{}-0.9.pkl".format(tp, ft,tp1)


        with open(ff, "rb") as f:
            paras = pk.load(f, encoding='latin1')


        for i in range(100):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(100,200):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(200,300):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(300,400):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(400,500):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(500,600):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(600,700):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(700,800):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(800,900):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(900,1000):
            label.append([i, 1])
            para.append(paras[i])

        para_list.append(np.array(para))
        para_list_neg.append(np.array(para_neg))


    feats[tp1] = para_list
    feats_neg[tp1] = para_list_neg

ft_name=['feat_embed1','feat_embed2','feat_post']
from sklearn.model_selection import train_test_split

results=[]
# k=[1,4,6]
k = [0,1,2]
# tps1=['edu-gender-2']
tps=['pokec','fb','pubmed']
for tp1 in tps:
    for tp2 in tps:
        # if tp1==tp2:
        #     continue
        for j in k:
            ft_=feats[tp1][j]
            ft_neg = feats_neg[tp1][j]
            ft2_=feats[tp2][j]
            ft2_neg = feats_neg[tp2][j]
            na=tp1+'-'+tp2+'-'+ft_name[j]

            if np.shape(ft_)[1]!=np.shape(ft2_)[1]:
                continue

            # print(np.shape(ft_))
            # print(np.shape(ft_neg))

            x_train_pos, x_test_pos, y_train_pos, y_test_pos  = train_test_split(ft_, label,test_size=0.3, random_state=42)
            x_train_neg, x_test_neg, y_train_neg, y_test_neg = train_test_split(ft_neg, label_neg, test_size=0.3, random_state=42)

            x_train = np.concatenate((x_train_pos, x_train_neg), axis=0)
            x_test = np.concatenate((x_test_pos, x_test_neg), axis=0)
            y_train = np.concatenate((y_train_pos,y_train_neg), axis=0)
            y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)

            x_train_pos2, x_test_pos2, y_train_pos2, y_test_pos2 = train_test_split(ft2_, label, test_size=0.3,
                                                                                random_state=42)
            x_train_neg2, x_test_neg2, y_train_neg2, y_test_neg2 = train_test_split(ft2_neg, label_neg, test_size=0.3,
                                                                                random_state=42)

            x_train2 = np.concatenate((x_train_pos2, x_train_neg2), axis=0)
            x_test2 = np.concatenate((x_test_pos2, x_test_neg2), axis=0)
            y_train2 = np.concatenate((y_train_pos2, y_train_neg2), axis=0)
            y_test2 = np.concatenate((y_test_pos2, y_test_neg2), axis=0)

            # X_train, X_test, y_train, y_test = train_test_split(sim_matrix, y_label, test_size=0.3, random_state=42)
            #
            # # ######################################################################

            res_dir='./pca-transfer/'

            from sklearn import metrics
            from sklearn.neural_network import MLPClassifier

            mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,max_iter=1500,early_stopping=True)

            # print(x_train,y_train[:,1])
            # print(np.shape(x_train), np.shape(y_train[:, 1]))
            mlp.fit(x_train, y_train[:,1])

            print("Training set score: %f" % mlp.score(x_train, y_train[:,1]))
            print("Test set score: %f" % mlp.score(x_test2, y_test2[:,1]))

            y_score = mlp.predict(x_test2)
            proba = mlp.predict_proba(x_test2)
            proba=np.amax(proba,axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))
            # report = metrics.classification_report(y_test2[:,1], y_score, labels=range(3), output_dict=True)
            #
            # out = "{}/{}-mlp-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test2[i][0]
                prob = proba[i]

                tst = [y_score[i], y_test2[i][1],prob, y_test2[i][0]]
                tsts.append(tst)
            name = ['y_score', 'y_test_grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-mlp-12-13-0.9.csv".format(res_dir,na))
            print(acc,recall,precision,f1,auc)

            results.append([na,acc,recall,precision,f1,auc])

            # # ######################################################################
            # # # ######################################################################

            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(random_state=0)
            lr.fit(x_train, y_train[:,1])


            print("Training set score: %f" % lr.score(x_train, y_train[:,1]))
            print("Test set score: %f" % lr.score(x_test2, y_test2[:,1]))

            y_score = lr.predict(x_test2)
            proba = lr.predict_proba(x_test2)
            proba = np.amax(proba, axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))

            # report=metrics.classification_report(y_test2[:,1], y_score, labels=range(3),output_dict=True)
            # # print(report)
            #
            # out = "{}/{}-lr-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test[i][0]
                prob=proba[i]

                tst = [y_score[i], y_test2[i][1], prob,y_test2[i][0]]
                tsts.append(tst)
            name = ['pred_label', 'grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-lr-12-13-0.9.csv".format(res_dir,na))
            print(acc, recall, precision, f1, auc)

            results.append([na,acc, recall, precision, f1, auc])

            # # ######################################################################
            # # # ######################################################################

            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(max_depth=150, random_state=0)
            rf.fit(x_train, y_train[:,1])

            print("Training set score: %f" % rf.score(x_train, y_train[:,1]))
            print("Test set score: %f" % rf.score(x_test2, y_test2[:,1]))

            y_score = rf.predict(x_test2)
            proba = rf.predict_proba(x_test2)
            proba = np.amax(proba, axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))

            # report=metrics.classification_report(y_test2[:,1], y_score, labels=range(3),output_dict=True)
            # # print(report)
            #
            # out = "{}/{}-rf-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test2[i][0]
                prob=proba[i]

                tst = [y_score[i], y_test2[i][1], prob,y_test2[i][0]]
                tsts.append(tst)
            name = ['pred_label', 'grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-rf-12-13-0.9.csv".format(res_dir,na))
            print(acc, recall, precision, f1, auc)

            results.append([na,acc, recall, precision, f1, auc])

print(results)
name = ['name','acc', 'recall', 'precision', 'f1', 'auc']
result = pd.DataFrame(columns=name, data=results)
result.to_csv("{}/results_transfer-0.9.csv".format(res_dir))



dataset=['pokec','fb','pubmed']
tps=['pokec-interintra','fb-interintra','pubmed-interintra']
fts=['embed1','embed2','post']
feats={}
feats_neg={}
i=0
for tp1 in tps:

    res_dir = './pca-tranfer/'
    tp = 'pca'
    para_list = []
    para_list_neg = []

    for ft in fts:
        label = []
        label_neg = []
        para = []
        para_neg = []

        ff = "./{}/{}-99-5-{}-0.9.pkl".format(tp, ft,tp1)


        with open(ff, "rb") as f:
            paras = pk.load(f, encoding='latin1')


        for i in range(100):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(100,200):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(200,300):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(300,400):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(400,500):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(500,600):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(600,700):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(700,800):
            label.append([i, 1])
            para.append(paras[i])
        for i in range(800,900):
            label_neg.append([i,0])
            para_neg.append(paras[i])
        for i in range(900,1000):
            label.append([i, 1])
            para.append(paras[i])

        para_list.append(np.array(para))
        para_list_neg.append(np.array(para_neg))


    feats[tp1] = para_list
    feats_neg[tp1] = para_list_neg

ft_name=['feat_embed1','feat_embed2','feat_post']
from sklearn.model_selection import train_test_split

results=[]
# k=[1,4,6]
k = [0,1,2]
# tps1=['edu-gender-2']
tps=['pokec-interintra','fb-interintra','pubmed-interintra']
for tp1 in tps:
    for tp2 in tps:
        # if tp1==tp2:
        #     continue
        for j in k:
            ft_=feats[tp1][j]
            ft_neg = feats_neg[tp1][j]
            ft2_=feats[tp2][j]
            ft2_neg = feats_neg[tp2][j]
            na=tp1+'-'+tp2+'-'+ft_name[j]

            if np.shape(ft_)[1]!=np.shape(ft2_)[1]:
                continue

            # print(np.shape(ft_))
            # print(np.shape(ft_neg))

            x_train_pos, x_test_pos, y_train_pos, y_test_pos  = train_test_split(ft_, label,test_size=0.3, random_state=42)
            x_train_neg, x_test_neg, y_train_neg, y_test_neg = train_test_split(ft_neg, label_neg, test_size=0.3, random_state=42)

            x_train = np.concatenate((x_train_pos, x_train_neg), axis=0)
            x_test = np.concatenate((x_test_pos, x_test_neg), axis=0)
            y_train = np.concatenate((y_train_pos,y_train_neg), axis=0)
            y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)

            x_train_pos2, x_test_pos2, y_train_pos2, y_test_pos2 = train_test_split(ft2_, label, test_size=0.3,
                                                                                random_state=42)
            x_train_neg2, x_test_neg2, y_train_neg2, y_test_neg2 = train_test_split(ft2_neg, label_neg, test_size=0.3,
                                                                                random_state=42)

            x_train2 = np.concatenate((x_train_pos2, x_train_neg2), axis=0)
            x_test2 = np.concatenate((x_test_pos2, x_test_neg2), axis=0)
            y_train2 = np.concatenate((y_train_pos2, y_train_neg2), axis=0)
            y_test2 = np.concatenate((y_test_pos2, y_test_neg2), axis=0)

            # X_train, X_test, y_train, y_test = train_test_split(sim_matrix, y_label, test_size=0.3, random_state=42)
            #
            # # ######################################################################

            res_dir='./pca-transfer/'

            from sklearn import metrics
            from sklearn.neural_network import MLPClassifier

            mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,max_iter=1500,early_stopping=True)

            # print(x_train,y_train[:,1])
            # print(np.shape(x_train), np.shape(y_train[:, 1]))
            mlp.fit(x_train, y_train[:,1])

            print("Training set score: %f" % mlp.score(x_train, y_train[:,1]))
            print("Test set score: %f" % mlp.score(x_test2, y_test2[:,1]))

            y_score = mlp.predict(x_test2)
            proba = mlp.predict_proba(x_test2)
            proba=np.amax(proba,axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))
            # report = metrics.classification_report(y_test2[:,1], y_score, labels=range(3), output_dict=True)
            #
            # out = "{}/{}-mlp-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test2[i][0]
                prob = proba[i]

                tst = [y_score[i], y_test2[i][1],prob, y_test2[i][0]]
                tsts.append(tst)
            name = ['y_score', 'y_test_grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-mlp-12-13-0.9.csv".format(res_dir,na))
            print(acc,recall,precision,f1,auc)

            results.append([na,acc,recall,precision,f1,auc])

            # # ######################################################################
            # # # ######################################################################

            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(random_state=0)
            lr.fit(x_train, y_train[:,1])


            print("Training set score: %f" % lr.score(x_train, y_train[:,1]))
            print("Test set score: %f" % lr.score(x_test2, y_test2[:,1]))

            y_score = lr.predict(x_test2)
            proba = lr.predict_proba(x_test2)
            proba = np.amax(proba, axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))

            # report=metrics.classification_report(y_test2[:,1], y_score, labels=range(3),output_dict=True)
            # # print(report)
            #
            # out = "{}/{}-lr-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test[i][0]
                prob=proba[i]

                tst = [y_score[i], y_test2[i][1], prob,y_test2[i][0]]
                tsts.append(tst)
            name = ['pred_label', 'grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-lr-12-13-0.9.csv".format(res_dir,na))
            print(acc, recall, precision, f1, auc)

            results.append([na,acc, recall, precision, f1, auc])

            # # ######################################################################
            # # # ######################################################################

            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(max_depth=150, random_state=0)
            rf.fit(x_train, y_train[:,1])

            print("Training set score: %f" % rf.score(x_train, y_train[:,1]))
            print("Test set score: %f" % rf.score(x_test2, y_test2[:,1]))

            y_score = rf.predict(x_test2)
            proba = rf.predict_proba(x_test2)
            proba = np.amax(proba, axis=1)
            print(metrics.f1_score(y_test2[:,1], y_score, average='micro'))
            print(metrics.classification_report(y_test2[:,1], y_score, labels=range(3)))

            # report=metrics.classification_report(y_test2[:,1], y_score, labels=range(3),output_dict=True)
            # # print(report)
            #
            # out = "{}/{}-rf-report-12-13.csv".format(res_dir, na)
            #
            # df = pd.DataFrame(report).transpose()
            # df.to_csv(out, index=True)

            acc = accuracy_score(y_test2[:,1],y_score)
            recall = recall_score(y_test2[:,1],y_score)
            precision = precision_score(y_test2[:,1],y_score)
            f1 = f1_score(y_test2[:,1],y_score)
            auc = roc_auc_score(y_test2[:,1], proba)


            tsts = []
            for i in range(len(y_score)):
                id_=y_test2[i][0]
                prob=proba[i]

                tst = [y_score[i], y_test2[i][1], prob,y_test2[i][0]]
                tsts.append(tst)
            name = ['pred_label', 'grd','prob', 'index']
            result = pd.DataFrame(columns=name, data=tsts)
            result.to_csv("{}/{}-rf-12-13-0.9.csv".format(res_dir,na))
            print(acc, recall, precision, f1, auc)

            results.append([na,acc, recall, precision, f1, auc])

print(results)
name = ['name','acc', 'recall', 'precision', 'f1', 'auc']
result = pd.DataFrame(columns=name, data=results)
result.to_csv("{}/results_transfer-interintra-0.9.csv".format(res_dir))