from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data_pokec_ruikai, accuracy
from models import GCN_pia2

from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import pandas as pd
import pickle

import os

# from keras.layers import Input, Dense
# from keras.models import Model

def evaluation(output, labels, name):
    preds = output.max(1)[1].type_as(labels)
    out = output.max(1)[0]
    # print(out,torch.max(output.data,1))
    # preds = torch.round(output)
    out=out.detach().cpu().numpy()
    y_pred = preds.detach().cpu().numpy()
    y_label = labels.detach().cpu().numpy()
    # print(y_pred,y_label)
    acc = accuracy_score(y_label, y_pred)
    recall = recall_score(y_label, y_pred)
    precision = precision_score(y_label, y_pred)
    f1 = f1_score(y_label, y_pred)
    auc = roc_auc_score(y_label, out)

    print('{} accuracy:'.format(name), acc,
          '{} f1:'.format(name), f1,
          '{} auc:'.format(name), auc)

    return [acc, recall, precision, f1, auc]



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

rnd=300
test_loss_acc=[]
train_loss_acc=[]
result_train=[]
result_test=[]
for sed in range(1,6):
    for ii in range(-100,100):
        seed=sed
        # Load data
        adj, features, labels, idx_train, idx_test = load_data_pokec_ruikai(ii,sed)

        # Model and optimizer
        model = GCN_pia2(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout)
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)

        if args.cuda:
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()
            idx_train = idx_train.cuda()
            # idx_val = idx_val.cuda()
            idx_test = idx_test.cuda()


        def train(epoch):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            output,embed1,embed2,embed3 = model(features, adj)
            # print(output[idx_train])
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            # print(labels[idx_train])
            # exit()
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if not args.fastmode:
                # Evaluate validation set performance separately,
                # deactivates dropout during validation run.
                model.eval()
                output,embed1,embed2,embed3 = model(features, adj)

            # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            # acc_val = accuracy(output[idx_val], labels[idx_val])



            if (epoch+1) % 100==0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      # 'loss_val: {:.4f}'.format(loss_val.item()),
                      # 'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))
            return loss_train.item(),acc_train.item(),output,embed1,embed2,embed3


        def test():
            model.eval()
            para={}
            cnt=0
            for p in model.parameters():
                # print(p)
                p = p.detach().cpu().numpy()
                # print(p)
                para[cnt]=p
                cnt+=1

            output,embed1,embed2,embed3 = model(features, adj)
            loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
            return output,embed1,embed2,embed3,loss_test.item(),acc_test.item(),para

        def save_model(net,seed):
            PATH = './combined-adj-feat-train-{}_net-12-13-gender-original.pth'.format(seed)
            torch.save(net.state_dict(), PATH)



        # Train model
        t_total = time.time()
        best_valid_loss = 99999999
        for epoch in range(args.epochs):
            loss_train, acc_train,output_train,embed1,embed2,embed3=train(epoch)

            patience = 50

            if loss_train < best_valid_loss:
                best_valid_loss = loss_train
                trail_count = 0
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join('tmp',f'gcn_best.pt'))

            else:
                trail_count += 1
                if trail_count > patience:
                    print(f'  Early Stop, the best Epoch is {best_epoch}, validation loss: {best_valid_loss:.4f}.')
                    break

        model.load_state_dict(torch.load(os.path.join('tmp',f'gcn_best.pt')))

        train_loss_acc.append([loss_train, acc_train])

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(output_train)
        # print(np.shape(output_train))
        # Testing
        output_test,embed1,embed2,embed3,loss_test, acc_test,para=test()
        # save_model(model,seed)
        test_loss_acc.append([loss_test, acc_test])

        eval_train = evaluation(output_train[idx_train], labels[idx_train],str(sed)+'-'+str(ii))
        eval_test = evaluation(output_test[idx_test], labels[idx_test],str(sed)+'-'+str(ii))


        result_train.append(eval_train)
        result_test.append(eval_test)


        emb_matrix1=embed1.detach().cpu().numpy()
        emb_matrix2 = embed2.detach().cpu().numpy()
        emb_matrix3 = embed3.detach().cpu().numpy()
        # print(emb_matrix)
        # print(np.shape(emb_matrix))
        output_train = output_train.detach().cpu().numpy()
        output_test = output_test.detach().cpu().numpy()

        # with open("./data/pokec-para-{}-{}-12-13-gender-original.pkl".format(str(ii), str(sed)), "wb") as f:
        #     pickle.dump(para, f)
        #

        with open('./pokec-small-3.16/pokec-embed1-%s-%s-12-13.txt'%(ii,seed),'w') as f:
            f.write('%d %d\n' % (np.shape(emb_matrix1)[0], args.hidden))
            for item in emb_matrix1:
                for jtem in item:
                    f.write(str(jtem) + '\t')
                f.write('\n')
            f.close()

        with open('./pokec-small-3.16/pokec-embed2-%s-%s-12-13.txt'%(ii,seed),'w') as f:
            f.write('%d %d\n' % (np.shape(emb_matrix2)[0], args.hidden))
            for item in emb_matrix2:
                for jtem in item:
                    f.write(str(jtem) + '\t')
                f.write('\n')
            f.close()

        with open('./pokec-small-3.16/pokec-embed3-%s-%s-12-13.txt'%(ii,seed),'w') as f:
            f.write('%d %d\n' % (np.shape(emb_matrix3)[0], args.hidden))
            for item in emb_matrix3:
                for jtem in item:
                    f.write(str(jtem) + '\t')
                f.write('\n')
            f.close()


        with open('./pokec-small-3.16/pokec-output_train-%s-%s-12-13.txt'%(ii,seed),'w') as f:
            for item in output_train:
                for jtem in item:
                    f.write(str(jtem) + '\t')
                f.write('\n')
            f.close()

        with open('./pokec-small-3.16/pokec-output_test-%s-%s-12-13.txt'%(ii,seed),'w') as f:
            for item in output_test:
                for jtem in item:
                    f.write(str(jtem) + '\t')
                f.write('\n')
            f.close()

        rnd += 1

        # if (rnd + 1) % 100 == 0:
        #     data1 = pd.DataFrame(result_train)
        #     data1.to_csv('result_train-gender-original-gender-%s.csv' % (rnd))
        #
        #     data2 = pd.DataFrame(result_test)
        #     data2.to_csv('result_test-gender-original-gender-%s.csv' % (rnd))


        # data1 = pd.DataFrame(result_train)
# data1.to_csv('result_train-gender-original.csv')
#
# data2 = pd.DataFrame(result_test)
# data2.to_csv('result_test-gender-original.csv')


    # In[ ]:



    # encoding_dim = 1
    #
    # input_img = Input(shape=(4,))
    #
    # encoded = Dense(encoding_dim, activation='relu')(input_img)
    # decoded = Dense(4, activation='sigmoid')(encoded)
    # autoencoder = Model(input_img, decoded)
    # encoder = Model(input_img, encoded)
    # encoded_input = Input(shape=(encoding_dim,))
    # decoder_layer = autoencoder.layers[-1]
    # decoder = Model(encoded_input, decoder_layer(encoded_input))
    #
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    #
    # autoencoder.fit(X, X,epochs=1000,batch_size=256,shuffle=True)
    # encoded_imgs = encoder.predict(X)
    # decoded_imgs=decoder.predict(encoded_imgs)
    # print('000',encoded_imgs.shape)
    # print(encoded_imgs)
    # print(decoded_imgs)
    #
    # #In[ ]:
    #
    #
    # # print(encoded_imgs)
    #
    #
    # # In[ ]:
    #
    #
    #
    # #######  with autoencoder
    # from sklearn.cluster import KMeans
    # from sklearn.metrics import accuracy_score
    # accuracy=[]
    # for i in range(100):
    #   kmeans = KMeans(n_clusters=2, random_state=i).fit(encoded_imgs)
    #   #kmeans = KMeans(n_clusters=2, random_state=i).fit(X)
    #   #print(kmeans.labels_)
    #   ylabel=[1]*train_emb.shape[0] + [0]*test_emb.shape[0]
    #   acc = accuracy_score(kmeans.labels_, ylabel)
    #   accuracy.append(acc)
    # print(max(accuracy))
    # print(np.shape(X))
    # print(np.shape(ylabel))
    #
    # cents=kmeans.cluster_centers_
    # # print('cents',cents)
    #
    # dist0=0
    # dist1=0
    #
    # for l in range(len(kmeans.labels_)):
    #     dist0=np.sqrt(np.sum(np.square(encoded_imgs[l] - cents[0])))
    #     dist1 = np.sqrt(np.sum(np.square(encoded_imgs[l] - cents[1])))
    #     if kmeans.labels_[l]==0 and (dist0<dist1):
    #         cent0= cents[0]
    #         cent1 = cents[1]
    #         break
    #     elif kmeans.labels_[l]==0 and (dist0>dist1):
    #         cent1 = cents[0]
    #         cent0 = cents[1]
    #         break
    # # print(l)
    #
    # dis0=[]
    # dis1=[]
    #
    # for l in range(len(kmeans.labels_)):
    #
    #     if kmeans.labels_[l]==0:
    #         dist = np.sqrt(np.sum(np.square(encoded_imgs[l] - cent0)))
    #         dis0.append(dist)
    #
    #     else:
    #         dist = np.sqrt(np.sum(np.square(encoded_imgs[l] - cent1)))
    #         dis1.append(dist)
    #
    # # print(cent0,cent1)
    # dist0=sum(dis0)/len(dis0)
    # dist1=sum(dis1)/len(dis1)
    #
    # print(dist0,dist1)
    #
    #
    #
    #
    # #######  no autoencoder
    # accuracy_noen=[]
    # for i in range(100):
    #   # kmeans = KMeans(n_clusters=2, random_state=i).fit(encoded_imgs)
    #   kmeans_noen = KMeans(n_clusters=2, random_state=i).fit(X)
    #   # print(kmeans_noen)
    #   # print(kmeans.labels_)
    #   ylabel_noen=[1]*train_emb.shape[0] + [0]*test_emb.shape[0]
    #   acc_noen = accuracy_score(kmeans_noen.labels_, ylabel_noen)
    #   accuracy_noen.append(acc_noen)
    # print(max(accuracy_noen))
    # print(np.shape(X))
    # print(np.shape(ylabel_noen))
    #
    #
    #
    # cents_noen=kmeans_noen.cluster_centers_
    # #print(cents_noen)
    #
    # dist0_noen=0
    # dist1_noen=0
    #
    # for l in range(len(kmeans_noen.labels_)):
    #     dist0_noen=np.sqrt(np.sum(np.square(X[l] - cents_noen[0])))
    #     dist1_noen = np.sqrt(np.sum(np.square(X[l] - cents_noen[1])))
    #     if kmeans_noen.labels_[l]==0 and (dist0_noen<dist1_noen):
    #         cent0_noen= cents_noen[0]
    #         cent1_noen = cents_noen[1]
    #         break
    #     elif kmeans_noen.labels_[l]==0 and (dist0_noen>dist1_noen):
    #         cent1_noen = cents_noen[0]
    #         cent0_noen = cents_noen[1]
    #         break
    #
    #
    # # print(l)
    #
    # dis0_noen=[]
    # dis1_noen=[]
    #
    # for l in range(len(kmeans_noen.labels_)):
    #
    #     if kmeans_noen.labels_[l]==0:
    #         dist_noen = np.sqrt(np.sum(np.square(X[l] - cent0_noen)))
    #         dis0_noen.append(dist_noen)
    #
    #     else:
    #         dist_noen = np.sqrt(np.sum(np.square(X[l] - cent1_noen)))
    #         dis1_noen.append(dist_noen)
    #
    # # print(cent0_noen,cent1_noen)
    # dist0_noen=sum(dis0_noen)/len(dis0_noen)
    # dist1_noen=sum(dis1_noen)/len(dis1_noen)
    # print(dist0_noen,dist1_noen)
    #
    # item=[dist0,dist1,len(dis0),len(dis1),dist0_noen,dist1_noen,len(dis0_noen),len(dis1_noen)]
    # print(item)
    #
    #
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, ylabel, test_size=0.3, random_state=42)
    #
    # # # ######################################################################
    #
    # from sklearn import metrics
    # from sklearn.neural_network import MLPClassifier
    #
    # mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64,32,16,18), random_state=1,max_iter=200)
    #
    # mlp.fit(X_train, y_train)
    #
    # print("Training set score: %f" % mlp.score(X_train, y_train))
    # print("Test set score: %f" % mlp.score(X_test, y_test))
    #
    # y_score = mlp.predict(X_test)
    # print(metrics.f1_score(y_test, y_score, average='micro'))
    # print(metrics.classification_report(y_test, y_score, labels=range(3)))
    #
    # # # ######################################################################
    #
    # from sklearn.ensemble import RandomForestClassifier
    #
    # rf = RandomForestClassifier(max_depth=150, random_state=0)
    # rf.fit(X_train, y_train)
    #
    # print("Training set score: %f" % rf.score(X_train, y_train))
    # print("Test set score: %f" % rf.score(X_test, y_test))
    #
    # y_score = rf.predict(X_test)
    # print(metrics.f1_score(y_test, y_score, average='micro'))
    # print(metrics.classification_report(y_test, y_score, labels=range(3)))
    #
    # # # ######################################################################
    #
    # from sklearn.multiclass import OneVsRestClassifier
    # from sklearn.svm import SVC
    #
    #
    # svm = OneVsRestClassifier(SVC())
    # svm.fit(X_train, y_train)
    #
    # print("Training set score: %f" % svm.score(X_train, y_train))
    # print("Test set score: %f" % svm.score(X_test, y_test))
    #
    # y_score = svm.predict(X_test)
    # print(metrics.f1_score(y_test, y_score, average='micro'))
    # print(metrics.classification_report(y_test, y_score, labels=range(3)))
    #
    #
    #
    #
    #
    #
    # # idx_test = np.concatenate((np.ones(3800), np.zeros(3824)), axis=0)
    # # idx_val = np.concatenate((np.zeros(3800), np.ones(1200)), axis=0)
    # # idx_val = np.concatenate((idx_val, np.zeros(2624)), axis=0)
    # # idx_train = np.concatenate((np.zeros(5000), np.ones(2624)), axis=0)
    #
    # import random
    # np.random.seed(1)
    # test_sample_index=random.sample(range(train_emb.shape[0],(train_emb.shape[0]+test_emb.shape[0])), 100)
    # #val_sample_index=random.sample(range(3800,5000), 100)
    # np.random.seed(1)
    # train_sample_index=random.sample(range(train_emb.shape[0]), 100)
    #
    # X_sample_index= np.concatenate((train_sample_index,test_sample_index), axis=0)
    # ylabels= np.concatenate((np.ones(np.shape(train_emb)[0]),np.zeros(np.shape(test_emb)[0])), axis=0)
    #
    # X_sample=X[X_sample_index]
    #
    #
    # y_sample=ylabels[X_sample_index]
    #
    #
    # decoded_imgs_sample=decoded_imgs[X_sample_index]
    #
    #
    # import matplotlib.pyplot as plt
    # from sklearn import manifold
    #
    #
    # def plot_embedding_2d(data, label, title):
    #     x_min, x_max = np.min(data, 0), np.max(data, 0)
    #     data = (data - x_min) / (x_max - x_min)
    #     fig = plt.figure()
    #     for i in range(data.shape[0]):
    #         plt.text(data[i, 0], data[i, 1], str(label[i]),
    #                  color=plt.cm.Set1(label[i]),
    #                  fontdict={'weight': 'bold', 'size': 5})
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.title(title)
    #     return fig
    #
    #
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    #
    # X_tsne = tsne.fit_transform(X_sample)
    # plot_embedding_2d(X_tsne,y_sample, "t-SNE 2D original embedding")
    # plt.show()
    #
    #
    # decoded_imgs_tsne=tsne.fit_transform(decoded_imgs_sample)
    # plot_embedding_2d(decoded_imgs_tsne,y_sample, "t-SNE 2D after decoder")
    # plt.show()
    #
    #
    #
    #
    # model =  manifold.TSNE(n_components=2)
    # node_pos = model.fit_transform(X_sample)
    #
    # color_idx = {}
    # for i in range(len(y_sample)):
    #     color_idx.setdefault(y_sample[i], [])
    #     color_idx[y_sample[i]].append(i)
    #
    # for c, idx in color_idx.items():
    #     if c==0:
    #
    #         if (len(dis0_noen)>len(dis1_noen)):
    #             cc='member'
    #             cl = 'red'
    #         else:
    #             cc='non-member'
    #             cl = 'blue'
    #     else:
    #
    #         if (len(dis0_noen)<len(dis1_noen)):
    #             cc='member'
    #             cl = 'red'
    #         else:
    #             cc='non-member'
    #             cl = 'blue'
    #
    #     plt.scatter(node_pos[idx, 0], node_pos[idx, 1],  s=10,c=cl,label=cc)
    # plt.legend()
    # plt.savefig(fname="gcn-fb-orig",format="pdf")
    # plt.savefig(fname="gcn-fb-orig",format="eps")
    #
    # plt.show()
    #
    #
    # model =  manifold.TSNE(n_components=2)
    # node_pos = model.fit_transform(decoded_imgs_sample)
    #
    # color_idx = {}
    # for i in range(len(y_sample)):
    #     color_idx.setdefault(y_sample[i], [])
    #     color_idx[y_sample[i]].append(i)
    #
    # for c, idx in color_idx.items():
    #     # print(c)
    #     # print(idx)
    #     if c==0:
    #
    #         if (len(dis0)>len(dis1)):
    #             cc='member'
    #             cl = 'red'
    #         else:
    #             cc='non-member'
    #             cl = 'blue'
    #     else:
    #
    #         if (len(dis0)<len(dis1)):
    #             cc='member'
    #             cl = 'red'
    #         else:
    #             cc='non-member'
    #             cl = 'blue'
    #     plt.scatter(node_pos[idx, 0], node_pos[idx, 1], s=10,c=cl,label=cc)
    # plt.legend()
    # plt.savefig(fname="gcn-fb-decode",format="pdf")
    # plt.savefig(fname="gcn-fb-decode",format="eps")
    # plt.show()
    #
    # print(item)
    #
