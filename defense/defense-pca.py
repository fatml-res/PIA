from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np


from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import pandas as pd
import pickle as pk

# from keras.layers import Input, Dense
# from keras.models import Model
from sklearn.decomposition import PCA


def readembds(file_name, dim):
    file = open(file_name)
    first_line = file.readline()
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
    pca11 = PCA(n_components=dim)
    pe11 = pca11.fit_transform(embeddings)
    embeddings = np.array(pe11)
    embeddings1 = np.amax(embeddings, axis=1)
    return embeddings, embeddings1




dataset = ['pokec', 'fb', 'pubmed']
tps=['fb','pokec','pubmed','pokec-interintra','pubmed-interintra','fb-interintra']
feats = {}
feats_neg = {}
i = 0
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
    feat_embed0 = {}
    feat_embed1 = {}
    feat_embed2 = {}
    feat_embed3 = {}
    feat_embed0_neg = {}
    feat_embed1_neg = {}
    feat_embed2_neg = {}
    feat_embed3_neg = {}

    if tp1 == 'fb':
        p11 = 12#number of components to save 99% of the original informantion, it should be pre-calcalated by using PCA
        p21 = 10#number of components to save 95% of the original informantion
        p31 = 10#number of components to save 90% of the original informantion
        p41 = 9#number of components to save 80% of the original informantion
        p51 = 8#number of components to save 70% of the original informantion
        p12 = 5
        p22 = 4
        p32 = 3
        p42 = 3
        p52 = 2

    elif tp1 == 'pokec':
        p11 = 10
        p21 = 8
        p31 = 8
        p41 = 7
        p51 = 6
        p12 = 8
        p22 = 7
        p32 = 6
        p42 = 6
        p52 = 6

    elif tp1 == 'pubmed':
        p11 = 8
        p21 = 7
        p31 = 6
        p41 = 6
        p51 = 5
        p12 = 6
        p22 = 6
        p32 = 5
        p42 = 5
        p52 = 4


    elif tp1 == 'fb-einterintra':
        p11 = 11
        p21 = 10
        p31 = 9
        p41 = 8
        p51 = 6
        p12 = 8
        p22 = 7
        p32 = 6
        p42 = 6
        p52 = 6

    elif tp1 == 'pubmed-interintra':
        p11 = 7
        p21 = 6
        p31 = 5
        p41 = 5
        p51 = 4
        p12 = 6
        p22 = 6
        p32 = 5
        p42 = 5
        p52 = 5

    elif tp1 == 'pokec-interintra':
        p11 = 10
        p21 = 8
        p31 = 8
        p41 = 6
        p51 = 5
        p12 = 14
        p22 = 10
        p32 = 9
        p42 = 8
        p52 = 7

    ps1 = [p11, p21, p31, p41, p51]
    ps2 = [p12, p22, p32, p42, p52]

    ps11 = list(set(ps1))
    ps22 = list(set(ps2))

    # print(ps11, ps22)

    for iii in range(len(ps1)):
        p_ = str(ps1[iii])
        feat_embed0_neg[p_] = []
        feat_embed1_neg[p_] = []
        feat_embed0[p_] = []
        feat_embed1[p_] = []

    for iii in range(len(ps2)):
        p_ = str(ps2[iii])
        feat_embed2[p_] = []
        feat_embed3[p_] = []

        feat_embed2_neg[p_] = []
        feat_embed3_neg[p_] = []


    f_fir=tp1


    for sed in range(1, 6):
        for ii in range(-100, 100): ##totally 1000 graphs, for ii in range(-100,0): negative graphs, for ii in range(0,100): positive graphs
            s = ii + 100
            seed = sed

            embeds0 = {}
            embeds1 = {}
            embeds2 = {}
            embeds3 = {}

            for ps in ps11:
                file1 = './%s/pokec-embed1-%s-%s-12-13-%s.txt' % (f_fir, ii, seed, tp1)
                embed0, embed1 = readembds(file1, ps)
                ps = str(ps)
                embeds0[ps] = embed0
                embeds1[ps] = embed1

            for ps in ps22:
                file2 = './%s/pokec-embed2-%s-%s-12-13-%s.txt' % (f_fir, ii, seed, tp1)
                embed2, embed3 = readembds(file2, ps)
                ps = str(ps)
                embeds2[ps] = embed2
                embeds3[ps] = embed3

            print(embeds0.keys())



            for iii in range(len(ps1)):
                p_ = str(ps1[iii])

                if (int(s / 100)) % 2 == 0:
                    feat_embed0_neg[p_].append(np.array(embeds0[p_]).flatten())
                    feat_embed1_neg[p_].append(np.array(embeds1[p_]).flatten())


                else:
                    feat_embed0[p_].append(np.array(embeds0[p_]).flatten())
                    feat_embed1[p_].append(np.array(embeds1[p_]).flatten())

                p_ = str(ps2[iii])

                if (int(s / 100)) % 2 == 0:
                    feat_embed2_neg[p_].append(np.array(embeds2[p_]).flatten())
                    feat_embed3_neg[p_].append(np.array(embeds3[p_]).flatten())
                else:
                    feat_embed2[p_].append(np.array(embeds2[p_]).flatten())
                    feat_embed3[p_].append(np.array(embeds3[p_]).flatten())

    results = []
    for jjj in range(len(ps1)):
        p_1 = str(ps1[jjj])
        p_2 = str(ps2[jjj])

        feats = dict()
        feats[0] = np.array(feat_embed0[p_1])
        feats[1] = np.array(feat_embed1[p_1])
        feats[2] = np.array(feat_embed2[p_2])
        feats[3] = np.array(feat_embed3[p_2])

        feats_neg = dict()
        feats_neg[0] = np.array(feat_embed0[p_1])
        feats_neg[1] = np.array(feat_embed1[p_1])
        feats_neg[2] = np.array(feat_embed2[p_2])
        feats_neg[3] = np.array(feat_embed3[p_2])

        for i in range(100):
            label_neg.append([i, 0])
        for i in range(100, 200):
            label.append([i, 1])
        for i in range(200, 300):
            label_neg.append([i, 0])
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

        ft_name = ['feat_emb0', 'feat_emb1', 'feat_emb2', 'feat_emb3']
        from sklearn.model_selection import train_test_split

        k = [0, 1, 2, 3]
        for j in k:
            ft_ = feats[j]
            ft_neg = feats_neg[j]
            na = ft_name[j]

            print(np.shape(ft_))

            x_train_pos, x_test_pos, y_train_pos, y_test_pos = train_test_split(ft_, label, test_size=0.3,
                                                                                random_state=42)
            x_train_neg, x_test_neg, y_train_neg, y_test_neg = train_test_split(ft_neg, label_neg, test_size=0.3,
                                                                                random_state=42)

            x_train = np.concatenate((x_train_pos, x_train_neg), axis=0)
            x_test = np.concatenate((x_test_pos, x_test_neg), axis=0)
            y_train = np.concatenate((y_train_pos, y_train_neg), axis=0)
            y_test = np.concatenate((y_test_pos, y_test_neg), axis=0)
            #
            # # ######################################################################

            res_dir = './pca-defense/'

            from sklearn import metrics
            from sklearn.neural_network import MLPClassifier

            mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,
                                max_iter=1500, early_stopping=True)

            print(x_train, y_train[:, 1])
            print(np.shape(x_train), np.shape(y_train[:, 1]))
            mlp.fit(x_train, y_train[:, 1])

            print("Training set score: %f" % mlp.score(x_train, y_train[:, 1]))
            print("Test set score: %f" % mlp.score(x_test, y_test[:, 1]))

            y_score = mlp.predict(x_test)
            proba = mlp.predict_proba(x_test)
            proba = np.amax(proba, axis=1)
            print(metrics.f1_score(y_test[:, 1], y_score, average='micro'))
            print(metrics.classification_report(y_test[:, 1], y_score, labels=range(3)))


            acc = accuracy_score(y_test[:, 1], y_score)
            recall = recall_score(y_test[:, 1], y_score)
            precision = precision_score(y_test[:, 1], y_score)
            f1 = f1_score(y_test[:, 1], y_score)
            auc = roc_auc_score(y_test[:, 1], proba)

            tsts = []
            for i in range(len(y_score)):
                id_ = y_test[i][0]
                prob = proba[i]

                tst = [y_score[i], y_test[i][1], prob, y_test[i][0]]
                tsts.append(tst)

            results.append([na, acc, recall, precision, f1, auc])

            # # ######################################################################
            # # # ######################################################################

            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(random_state=0)
            lr.fit(x_train, y_train[:, 1])

            print("Training set score: %f" % lr.score(x_train, y_train[:, 1]))
            print("Test set score: %f" % lr.score(x_test, y_test[:, 1]))

            y_score = lr.predict(x_test)
            proba = lr.predict_proba(x_test)
            proba = np.amax(proba, axis=1)
            print(metrics.f1_score(y_test[:, 1], y_score, average='micro'))
            print(metrics.classification_report(y_test[:, 1], y_score, labels=range(3)))


            acc = accuracy_score(y_test[:, 1], y_score)
            recall = recall_score(y_test[:, 1], y_score)
            precision = precision_score(y_test[:, 1], y_score)
            f1 = f1_score(y_test[:, 1], y_score)
            auc = roc_auc_score(y_test[:, 1], proba)

            tsts = []
            for i in range(len(y_score)):
                id_ = y_test[i][0]
                prob = proba[i]

                tst = [y_score[i], y_test[i][1], prob, y_test[i][0]]
                tsts.append(tst)

            results.append([na, acc, recall, precision, f1, auc])

            # # ######################################################################
            # # # ######################################################################

            from sklearn.ensemble import RandomForestClassifier

            rf = RandomForestClassifier(max_depth=150, random_state=0)
            rf.fit(x_train, y_train[:, 1])

            print("Training set score: %f" % rf.score(x_train, y_train[:, 1]))
            print("Test set score: %f" % rf.score(x_test, y_test[:, 1]))

            y_score = rf.predict(x_test)
            proba = rf.predict_proba(x_test)
            proba = np.amax(proba, axis=1)
            print(metrics.f1_score(y_test[:, 1], y_score, average='micro'))
            print(metrics.classification_report(y_test[:, 1], y_score, labels=range(3)))


            acc = accuracy_score(y_test[:, 1], y_score)
            recall = recall_score(y_test[:, 1], y_score)
            precision = precision_score(y_test[:, 1], y_score)
            f1 = f1_score(y_test[:, 1], y_score)
            auc = roc_auc_score(y_test[:, 1], proba)

            tsts = []
            for i in range(len(y_score)):
                id_ = y_test[i][0]
                prob = proba[i]

                tst = [y_score[i], y_test[i][1], prob, y_test[i][0]]
                tsts.append(tst)


            results.append([na, acc, recall, precision, f1, auc])

    print(results)
    name = ['na', 'pred_label', 'grd', 'prob', 'index']

    a = pd.DataFrame(columns=name, data=results)
    a.to_csv("{}/results-12-13-{}-{}.csv".format(res_dir, tp1, jjj))







