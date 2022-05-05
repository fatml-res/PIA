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
    return embeddings


tps=['pokec','fb','pubmed']
feats={}
feats_neg={}
i=0
results_mlp = []
results_lr = []
results_rf = []
tsts_mlp = []
tsts_lr = []
tsts_rf = []
for tp0 in tps:
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

            f_fir=tp0


            file1 = './%s/embed1-%s-%s-12-13-%s.txt' % (f_fir, ii, seed, tp0)
            embed1 = readembds(file1)
                

            file2 = './%s/embed2-%s-%s-12-13-%s.txt' % (f_fir, ii, seed, tp0)
            embed2 = readembds(file2)
          
        
            file3 = './%s/output_test-%s-%s-12-13-%s.txt' % (f_fir, ii, seed, tp0)
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

    # paras1 = np.concatenate((feat_para1, feat_para1_neg), axis=0)
    # paras2 = np.concatenate((feat_para2, feat_para2_neg), axis=0)
    # paras12 = np.concatenate((feat_para12, feat_para12_neg), axis=0)
    embed1 = np.concatenate((feat_embed1, feat_embed1_neg), axis=0)
    embed2 = np.concatenate((feat_embed2, feat_embed2_neg), axis=0)

    posts = np.concatenate((feat_post, feat_post_neg), axis=0)

    from keras.layers import Input, Dense, Dropout
    from keras.models import Model
    from keras.models import *
    from keras.layers import *
    from keras.callbacks import *

    tp = 'encode2'

    encoding_dim = 128

    paras1 = embed1

    encoding_dim1 = 512
    encoding_dim2 = 256
    encoding_dim3 = 128



    def encoder(input_img):

        encoded1 = Dense(encoding_dim1, activation='relu')(input_img)
        encoded1 = Dropout(0.1)(encoded1)

        encoded2 = Dense(encoding_dim2, activation='relu')(encoded1)
        encoded2 = Dropout(0.1)(encoded2)

        encoded= Dense(encoding_dim3, activation='relu')(encoded2)
        encoded= Dropout(0.1)(encoded)

        return encoded

    # decoded = Dense(dim, activation='sigmoid')(encoded)

    def decoder(encoded):
        decoded1 = Dense(encoding_dim2)(encoded)
        decoded2 = Dense(encoding_dim1)(decoded1)
        decoded = Dense(dim)(decoded2)
        return decoded


    dim = np.shape(paras1)[1]

    input_img = Input(shape=(dim,))

    autoencoder = Model(input_img, decoder(encoder(input_img)))
    encoderd = Model(input_img, encoder(input_img))
    encoded_input = Input(shape=(encoding_dim,))

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   verbose=0,
                                   mode='min',
                                   )
    check_point = ModelCheckpoint("models/liver-auto.h5",
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='min',
                                  )
    callbacks = [check_point, early_stopping]


    autoencoder.fit(paras1, paras1, epochs=1000, batch_size=1000, shuffle=True,validation_data=(paras1,paras1),callbacks=callbacks)
    encoded_imgs = encoderd.predict(paras1)

    with open("./{}/embed1-{}-{}-12-13-{}.pkl".format(tp, str(ii), str(sed),tp0), "wb") as f:
        pk.dump(encoded_imgs, f)

    # In[ ]:

    encoding_dim = 128

    paras1 = embed1

    encoding_dim1 = 512
    encoding_dim2 = 256
    encoding_dim3 = 128

    def encoder(input_img):

        encoded1 = Dense(encoding_dim1, activation='relu')(input_img)
        encoded1 = Dropout(0.1)(encoded1)

        encoded2 = Dense(encoding_dim2, activation='relu')(encoded1)
        encoded2 = Dropout(0.1)(encoded2)

        encoded= Dense(encoding_dim3, activation='relu')(encoded2)
        encoded= Dropout(0.1)(encoded)

        return encoded


    def decoder(encoded):
        decoded1 = Dense(encoding_dim2)(encoded)
        decoded2 = Dense(encoding_dim1)(decoded1)
        decoded = Dense(dim)(decoded2)
        return decoded


    dim = np.shape(paras1)[1]

    input_img = Input(shape=(dim,))

    autoencoder = Model(input_img, decoder(encoder(input_img)))
    encoderd = Model(input_img, encoder(input_img))
    encoded_input = Input(shape=(encoding_dim,))


    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   verbose=0,
                                   mode='min',
                                   )
    check_point = ModelCheckpoint("models/liver-auto.h5",
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='min',
                                  )
    callbacks = [check_point, early_stopping]


    autoencoder.fit(paras1, paras1, epochs=1000, batch_size=1000, shuffle=True,validation_data=(paras1,paras1),callbacks=callbacks)
    encoded_imgs = encoderd.predict(paras1)
    with open("./{}/embed2-{}-{}-12-13-{}.pkl".format(tp, str(ii), str(sed),tp1), "wb") as f:
        pk.dump(encoded_imgs, f)


    # In[ ]:

    encoding_dim = 128

    paras1 = embed1

    encoding_dim1 = 512
    encoding_dim2 = 256
    encoding_dim3 = 128


    def encoder(input_img):

        encoded1 = Dense(encoding_dim1, activation='relu')(input_img)
        encoded1 = Dropout(0.1)(encoded1)

        encoded2 = Dense(encoding_dim2, activation='relu')(encoded1)
        encoded2 = Dropout(0.1)(encoded2)

        encoded= Dense(encoding_dim3, activation='relu')(encoded2)
        encoded= Dropout(0.1)(encoded)

        return encoded

    def decoder(encoded):
        decoded1 = Dense(encoding_dim2)(encoded)
        decoded2 = Dense(encoding_dim1)(decoded1)
        decoded = Dense(dim)(decoded2)
        return decoded


    dim = np.shape(paras1)[1]

    input_img = Input(shape=(dim,))

    autoencoder = Model(input_img, decoder(encoder(input_img)))
    encoderd = Model(input_img, encoder(input_img))
    encoded_input = Input(shape=(encoding_dim,))


    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   verbose=0,
                                   mode='min',
                                   )
    check_point = ModelCheckpoint("models/liver-auto.h5",
                                  monitor='val_loss',
                                  verbose=1,
                                  save_best_only=True,
                                  mode='min',
                                  )
    callbacks = [check_point, early_stopping]


    autoencoder.fit(paras1, paras1, epochs=1000, batch_size=1000, shuffle=True,validation_data=(paras1,paras1),callbacks=callbacks)
    encoded_imgs = encoderd.predict(paras1)
    with open("./{}/encoder-{}-{}-{}.pkl".format(tp, str(ii), str(sed),tp0), "wb") as f:
        pk.dump(encoded_imgs, f)


