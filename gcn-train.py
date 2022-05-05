from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils2 import *
from models import *

from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import pandas as pd
import pickle

import math

# from keras.layers import Input, Dense
# from keras.models import Model


def stable_sigmoid(x):

    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(x)
        sig = z / (1 + z)
        return sig

def evaluation(output, labels, name):
    preds = output.max(1)[1].type_as(labels)
    out = output.max(1)[0]
    # print(out,torch.max(output.data,1))
    # preds = torch.round(output)
    out=out.detach().numpy()
    y_pred = preds.detach().numpy()
    y_label = labels.detach().numpy()
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



def evaluation2(output, labels, name):
    out_=output.detach().numpy()
    logits_ = []
    for i in out_:
        logit = [stable_sigmoid(i[0]) / (stable_sigmoid(i[0]) + stable_sigmoid(i[1]) + stable_sigmoid(i[2])),
                 stable_sigmoid(i[1]) / (stable_sigmoid(i[0]) + stable_sigmoid(i[1]) + stable_sigmoid(i[2])),
                 stable_sigmoid(i[2]) / (stable_sigmoid(i[0]) + stable_sigmoid(i[1]) + stable_sigmoid(i[2]))]
        logits_.append(logit)
    # print(logits_)
    logits_ = np.array(logits_)
    preds = output.max(1)[1].type_as(labels)
    out = output.max(1)[0]
    # print(out,torch.max(output.data,1))
    # preds = torch.round(output)
    out=out.detach().numpy()
    y_pred = preds.detach().numpy()
    y_label = labels.detach().numpy()
    # print(y_pred,y_label)
    acc = accuracy_score(y_label, y_pred)
    recall = recall_score(y_label, y_pred, average='macro')
    precision = precision_score(y_label, y_pred, average='macro')
    f1 = f1_score(y_label, y_pred, average='macro')
    auc = roc_auc_score(y_label, logits_, average='macro', multi_class="ovr")

    print('{} accuracy:'.format(name), acc,
          '{} f1:'.format(name), f1,
          '{} auc:'.format(name), auc)

    return [acc, recall, precision, f1, auc]





# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='pokec',
                    help='One of {pokec,fb,pubmed,pokec-interintra,fb-interintra,pubmed-interintra}')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

test_loss_acc=[]
train_loss_acc=[]
result_train=[]
result_test=[]
for sed in range(1,6): ##totally 1000 graphs, for ii in range(-100,0): negative graphs, for ii in range(0,100): positive graphs
    for ii in range(-100,100):
        seed=sed
        # Load data
        if args.dataset=='pokec':
            adj, features, labels, idx_train, idx_test = load_data_pokec(ii,sed)
        elif args.dataset == 'fb':
            adj, features, labels, idx_train, idx_test = load_data_fb(ii, sed)

        elif args.dataset == 'pubmed':
            adj, features, labels, idx_train, idx_test = load_data_pubmed(ii, sed)

        elif args.dataset == 'pokec-interintra':
            adj, features, labels, idx_train, idx_test = load_data_pokec_interintra(ii, sed)
        elif args.dataset == 'fb-interintra':
            adj, features, labels, idx_train, idx_test = load_data_fb_interintra(ii, sed)

        elif args.dataset == 'pubmed-interintra':
            adj, features, labels, idx_train, idx_test = load_data_pubmed_interintra(ii, sed)
        # Model and optimizer
        model = GCN_pia(nfeat=features.shape[1],
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
            output,embed1,embed2 = model(features, adj)
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
                output,embed1,embed2 = model(features, adj)

            # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            # acc_val = accuracy(output[idx_val], labels[idx_val])
            if (epoch+1) % 100==0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      # 'loss_val: {:.4f}'.format(loss_val.item()),
                      # 'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))
            return loss_train.item(),acc_train.item(),output,embed1,embed2


        def test():
            model.eval()
            para={}
            cnt=0
            for p in model.parameters():
                # print(p)
                p = p.detach().numpy()
                # print(p)
                para[cnt]=p
                cnt+=1

            output,embed1,embed2 = model(features, adj)
            loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
            return output,embed1,embed2,loss_test.item(),acc_test.item(),para

        def save_model(net,seed):
            PATH = './gender-original.pth'
            torch.save(net.state_dict(), PATH)



        # Train model
        t_total = time.time()
        for epoch in range(args.epochs):
            loss_train, acc_train,output_train,embed1,embed2=train(epoch)
        train_loss_acc.append([loss_train, acc_train])

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print(output_train)

        # Testing
        output_test,embed1,embed2,loss_test, acc_test,para=test()
        # save_model(model,seed)
        test_loss_acc.append([loss_test, acc_test])

        if args.dataset=='pubmed' or args.dataset=='pubmed-interintra':
            eval_train = evaluation2(output_train[idx_train], labels[idx_train], str(sed) + '-' + str(ii))
            eval_test = evaluation2(output_test[idx_test], labels[idx_test], str(sed) + '-' + str(ii))

        else:

            eval_train = evaluation(output_train[idx_train], labels[idx_train],str(sed)+'-'+str(ii))
            eval_test = evaluation(output_test[idx_test], labels[idx_test],str(sed)+'-'+str(ii))


        result_train.append(eval_train)
        result_test.append(eval_test)


        emb_matrix1=embed1.detach().numpy()
        emb_matrix2 = embed2.detach().numpy()

        output_train = output_train.detach().numpy()
        output_test = output_test.detach().numpy()


        tp=args.dataset

        with open("./{}/para-{}-{}-{}.pkl".format(tp,str(ii), str(sed),tp), "wb") as f:
            pickle.dump(para, f)


        with open('./{}/embed1-{}-{}-{}.txt'.format(tp,str(ii), str(sed),tp),'w') as f:
            f.write('%d %d\n' % (np.shape(emb_matrix1)[0], args.hidden))
            for item in emb_matrix1:
                for jtem in item:
                    f.write(str(jtem) + '\t')
                f.write('\n')
            f.close()

        with open('./{}/embed2-{}-{}-{}.txt'.format(tp,str(ii), str(sed),tp),'w') as f:
            f.write('%d %d\n' % (np.shape(emb_matrix2)[0], args.hidden))
            for item in emb_matrix2:
                for jtem in item:
                    f.write(str(jtem) + '\t')
                f.write('\n')
            f.close()


        with open('./{}/output_train-{}-{}-{}.txt'.format(tp,str(ii), str(sed),tp),'w') as f:
            for item in output_train:
                for jtem in item:
                    f.write(str(jtem) + '\t')
                f.write('\n')
            f.close()

        with open('./{}/output_test-{}-{}-{}.txt'.format(tp,str(ii), str(sed),tp),'w') as f:
            for item in output_test:
                for jtem in item:
                    f.write(str(jtem) + '\t')
                f.write('\n')
            f.close()


