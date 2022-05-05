from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

# import torch
# import torch.nn.functional as F
# import torch.optim as optim
#
# from utils import load_data_pokec, accuracy
# from models import GCN_pia

from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import pandas as pd
import pickle as pk

import random

# from keras.layers import Input, Dense
# from keras.models import Model

def readembds(file_name,ns):
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

    num_idx = int(ns * np.shape(embeddings)[1])
    idxs=list(range(np.shape(embeddings)[1]))

    embeddings_=[]

    for ebd in embeddings:

        idx_ = np.array(random.sample(idxs, num_idx))

        embeddings_.append(ebd[idx_])

    embeddings = embeddings_

    embeddings4 = np.amax(embeddings, axis=1)

    return embeddings,embeddings4


results_all=[]

tps=['fb','pokec','pubmed','pokec-interintra','pubmed-interintra','fb-interintra']
truncate_scales=[0.5,0.6,0.7,0.8,0.9,0,95,0.99]
res_dir1 = 'post-truncate-random'


for tp in tps:
   

    results = []
    for ns in truncate_scales:

        label=[]
        feat_embed0_neg=[]
        feat_embed1_neg = []

        feat_embed0=[]
        feat_embed1 = []

        feat_embed2_neg = []
        feat_embed3_neg = []

        feat_embed2 = []
        feat_embed3 = []

        label_neg=[]

        s=0


        for sed in range(1,6):
            for ii in range(-100,100): ##totally 1000 graphs, for ii in range(-100,0): negative graphs, for ii in range(0,100): positive graphs
                seed=sed
                seed=sed

                f = tp

                file1='./%s/pokec-embed1-%s-%s-12-13-%s.txt' % (f,ii, seed,tp)
               
                embed0,embed1 = readembds(file1,ns)

                file2='./%s/pokec-embed2-%s-%s-12-13-%s.txt' % (f,ii, seed,tp)
                embed2,embed3 = readembds(file2,ns)

                with open('./%s/pokec-embed1-%s-%s-%s-%s.txt' % (tp, ii, seed, tp,ns), 'w') as f:
                    f.write('%d %d\n' % (np.shape(embed0)[0], np.shape(embed0)[1]))
                    for item in embed0:
                        for jtem in item:
                            f.write(str(jtem) + '\t')
                        f.write('\n')
                    f.close()

                with open('./%s/pokec-embed2-%s-%s-%s-%s.txt' % (tp, ii, seed, tp,ns), 'w') as f:
                    f.write('%d %d\n' % (np.shape(embed2)[0], np.shape(embed2)[1]))
                    for item in embed2:
                        for jtem in item:
                            f.write(str(jtem) + '\t')
                        f.write('\n')
                    f.close()

                if (int(s / 100)) % 2 == 0:
                    feat_embed0_neg.append(np.array(embed0).flatten())
                    feat_embed1_neg.append(np.array(embed1).flatten())
                    feat_embed2_neg.append(np.array(embed2).flatten())
                    feat_embed3_neg.append(np.array(embed3).flatten())

                else:
                    feat_embed0.append(np.array(embed0).flatten())
                    feat_embed1.append(np.array(embed1).flatten())
                    feat_embed2.append(np.array(embed2).flatten())
                    feat_embed3.append(np.array(embed3).flatten())


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
        feats[0]=np.array(feat_embed0)
        feats[1]=np.array(feat_embed1)
        feats[2]=np.array(feat_embed2)
        feats[3]=np.array(feat_embed3)


        feats_neg=dict()
        feats_neg[0]=np.array(feat_embed0_neg)
        feats_neg[1]=np.array(feat_embed1_neg)
        feats_neg[2]=np.array(feat_embed2_neg)
        feats_neg[3]=np.array(feat_embed3_neg)

        ft_name=['feat_embed1','feat_embed2','feat_embed3','feat_embed4']

        ns=str(ns)
        from sklearn.model_selection import train_test_split

        k=[0,1,2,3]
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

            res_dir='./post-truncate-random/'

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
            result.to_csv("{}/{}-mlp-12-13-{}-{}.csv".format(res_dir,na,tp,ns))
            print(acc,recall,precision,f1,auc)

            nm=tp+na+ns

            results.append([nm,acc,recall,precision,f1,auc])
            results_all.append([nm, acc, recall, precision, f1, auc])

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
            result.to_csv("{}/{}-lr-12-13-{}-{}.csv".format(res_dir,na,tp,ns))
            print(acc, recall, precision, f1, auc)

            nm = tp + na + ns

            results.append([nm, acc, recall, precision, f1, auc])
            results_all.append([nm, acc, recall, precision, f1, auc])

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
            result.to_csv("{}/{}-rf-12-13-{}-{}.csv".format(res_dir,na,tp,ns))
            print(acc, recall, precision, f1, auc)

            nm = tp + na + ns

            results.append([nm, acc, recall, precision, f1, auc])
            results_all.append([nm, acc, recall, precision, f1, auc])

    print(results)
    name = ['tp','acc', 'recall', 'precision', 'f1', 'auc']
    result = pd.DataFrame(columns=name, data=results)
    result.to_csv("{}/results_{}-{}.csv".format(res_dir,tp,ns))


print(results)
name = ['tp','acc', 'recall', 'precision', 'f1', 'auc']
result = pd.DataFrame(columns=name, data=results_all)
result.to_csv("{}/results_truncate.csv".format(res_dir))